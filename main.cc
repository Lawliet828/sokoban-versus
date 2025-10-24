#include <drogon/drogon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
void addCorsHeaders(const drogon::HttpResponsePtr &resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
}

// 方向: 0-上, 1-下, 2-左, 3-右
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

// 时间限制（毫秒）
const int TIME_LIMIT_MS = 250;
constexpr int NEG_INF = std::numeric_limits<int>::min() / 8;
constexpr int OSCILLATION_PENALTY = 1200;

inline bool timeExceeded(const std::chrono::steady_clock::time_point &start,
                         int guard = 5) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    return elapsed.count() >= (TIME_LIMIT_MS - guard);
}

struct Position {
    int x, y;
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Position &other) const {
        return x == other.x && y == other.y;
    }
    bool operator<(const Position &other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

struct Move {
    Position person;
    int direction;
    int score;
    Move(Position p = Position(), int d = 0, int s = 0)
        : person(p), direction(d), score(s) {}
};

inline int encodePositionKey(int x, int y) {
    return ((x & 0xFFFF) << 16) ^ (y & 0xFFFF);
}

inline int encodePositionKey(const Position &p) {
    return encodePositionKey(p.x, p.y);
}

struct MovementMemory {
    std::unordered_map<int, int> lastDirection;
};

static std::unordered_map<std::string, MovementMemory> g_movementMemory;

// 游戏状态（用于哈希和重复检测）
struct GameState {
    std::vector<Position> persons;
    std::vector<Position> boxes;

    bool operator==(const GameState &other) const {
        return persons == other.persons && boxes == other.boxes;
    }

    size_t hash() const {
        size_t h = 0;
        for (const auto &p : persons) {
            h ^= std::hash<int>()(p.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(p.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        for (const auto &b : boxes) {
            h ^= std::hash<int>()(b.x * 997) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(b.y * 997) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

int manhattanDistance(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

int distanceToNearestGoalRow(const std::vector<std::vector<int>> &map, int side,
                             int rowCount, int columnCount, int currentRow) {
    int target_col = (side == 1) ? columnCount - 2 : 1;
    int best = std::numeric_limits<int>::max();
    for (int r = 0; r < rowCount; ++r) {
        if (map[r][target_col] == 0) {
            best = std::min(best, std::abs(r - currentRow));
        }
    }
    return best;
}

inline bool isStaticObstacle(int value) { return value == 3 || value == 4; }

bool goalDirectionBlocked(const std::vector<std::vector<int>> &map, int side,
                          const Position &boxPos, int row, int column) {
    (void)row;
    int offset = (side == 1) ? 1 : -1;
    int nextY = boxPos.y + offset;
    if (nextY < 0 || nextY >= column) {
        return true;
    }
    return isStaticObstacle(map[boxPos.x][nextY]);
}

bool opponentBlocksGoalDirection(const std::vector<std::vector<int>> &map,
                                 int side, const Position &boxPos, int row,
                                 int column) {
    (void)row;
    int opponent = (side == 1) ? 2 : 1;
    int offset = (side == 1) ? 1 : -1;
    int guardY = boxPos.y + offset;
    if (guardY < 0 || guardY >= column) {
        return false;
    }
    return map[boxPos.x][guardY] == opponent;
}

bool opponentAnchorsGoalDirection(const std::vector<std::vector<int>> &map,
                                  int side, const Position &boxPos, int row,
                                  int column) {
    if (!opponentBlocksGoalDirection(map, side, boxPos, row, column)) {
        return false;
    }
    int offset = (side == 1) ? 1 : -1;
    int guardY = boxPos.y + offset;
    int behindY = guardY + offset;
    if (behindY < 0 || behindY >= column) {
        return true;
    }
    return map[boxPos.x][behindY] != 0;
}

bool verticallyTrapped(const std::vector<std::vector<int>> &map,
                       const Position &boxPos, int row, int column) {
    (void)column;
    int upX = boxPos.x - 1;
    bool upBlocked = (upX < 0) || isStaticObstacle(map[upX][boxPos.y]);

    int downX = boxPos.x + 1;
    bool downBlocked = (downX >= row) || isStaticObstacle(map[downX][boxPos.y]);

    return upBlocked && downBlocked;
}

inline bool opponentOpposesPush(const std::vector<std::vector<int>> &map,
                                int side, const Position &person,
                                int direction, int row, int column) {
    int opponent = (side == 1) ? 2 : 1;
    int nx = person.x + dx[direction];
    int ny = person.y + dy[direction];
    if (nx < 0 || nx >= row || ny < 0 || ny >= column) {
        return false;
    }
    if (map[nx][ny] != 3) {
        return false;
    }
    const int opposeX = nx + dx[direction];
    const int opposeY = ny + dy[direction];
    if (opposeX < 0 || opposeX >= row || opposeY < 0 || opposeY >= column) {
        return false;
    }
    if (map[opposeX][opposeY] != opponent) {
        return false;
    }

    const int behindX = opposeX + dx[direction];
    const int behindY = opposeY + dy[direction];
    if (behindX < 0 || behindX >= row || behindY < 0 || behindY >= column) {
        return false;
    }
    if (map[behindX][behindY] != 0) {
        return false;
    }

    return true;
}

bool opponentMayPushInto(const std::vector<std::vector<int>> &map, int side,
                         const Position &target, int row, int column) {
    int opponent = (side == 1) ? 2 : 1;
    if (target.x < 0 || target.x >= row || target.y < 0 || target.y >= column) {
        return false;
    }
    if (map[target.x][target.y] != 0) {
        return false;
    }
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (map[i][j] != opponent) {
                continue;
            }
            for (int dir = 0; dir < 4; ++dir) {
                int bx = i + dx[dir];
                int by = j + dy[dir];
                int tx = bx + dx[dir];
                int ty = by + dy[dir];
                if (bx < 0 || bx >= row || by < 0 || by >= column ||
                    tx < 0 || tx >= row || ty < 0 || ty >= column) {
                    continue;
                }
                if (map[bx][by] == 3 && tx == target.x && ty == target.y &&
                    map[tx][ty] == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

// 获取当前游戏状态
GameState getCurrentState(const std::vector<std::vector<int>> &map, int side,
                         int row, int column) {
    GameState state;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            if (map[i][j] == side) {
                state.persons.push_back(Position(i, j));
            } else if (map[i][j] == 3) {
                state.boxes.push_back(Position(i, j));
            }
        }
    }
    std::sort(state.persons.begin(), state.persons.end());
    std::sort(state.boxes.begin(), state.boxes.end());
    return state;
}

bool isDeadlock(const std::vector<std::vector<int>> &map, Position box,
                int side, int row, int column) {
    int x = box.x, y = box.y;

    // 如果已经在目标区域，不是死局
    if ((side == 1 && y >= column - 2) || (side == 2 && y <= 1)) {
        return false;
    }

    // 检查是否贴在角落（两个方向都被墙/障碍物挡住）
    bool blocked_horizontal = false;
    bool blocked_vertical = false;

    // 横向检查
    if ((y == 1 || map[x][y - 1] == 4) &&
        (y == column - 2 || map[x][y + 1] == 4)) {
        blocked_horizontal = true;
    }

    // 纵向检查
    if ((x == 1 || map[x - 1][y] == 4) &&
        (x == row - 2 || map[x + 1][y] == 4)) {
        blocked_vertical = true;
    }

    // 两个方向都被堵死
    if (blocked_horizontal && blocked_vertical) {
        return true;
    }

    // 检查是否在边角（L型死角）
    int wall_count = 0;
    for (int dir = 0; dir < 4; dir++) {
        int nx = x + dx[dir];
        int ny = y + dy[dir];
        if (nx <= 0 || nx >= row - 1 || ny <= 0 || ny >= column - 1 ||
            map[nx][ny] == 4) {
            wall_count++;
        }
    }

    // 如果三面或以上被墙围住，很可能是死局
    if (wall_count >= 3) {
        return true;
    }

    return false;
}

// 计算启发式值：所有箱子到目标的最小距离和
int calculateHeuristic(const std::vector<std::vector<int>> &map, int side,
                      const std::vector<Position> &boxes, int row, int column) {
    int h = 0;
    int target_col = (side == 1) ? column - 2 : 1;

    for (const auto &box : boxes) {
        int dist = std::abs(box.y - target_col);
        h += dist * 10;

        if (isDeadlock(map, box, side, row, column)) {
            h += 1000;
        }
    }

    return h;
}

// 找到最优的箱子（综合考虑距离和位置）
Position findBestBox(const std::vector<std::vector<int>> &map, int side,
                     Position person, int row, int column) {
    Position best(-1, -1);
    int bestScore = -10000;
    int target_col = (side == 1) ? column - 2 : 1;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            if (map[i][j] == 3) {
                // 已经到目的地的箱子跳过
                if ((side == 1 && j >= target_col) ||
                    (side == 2 && j <= target_col)) {
                    continue;
                }
                Position boxPos(i, j);
                int score = 0;

                if (side == 1) {
                    score += j * 10;
                    if (j + 1 < column && map[i][j + 1] != 4 &&
                        map[i][j + 1] != 1 && map[i][j + 1] != 2 &&
                        map[i][j + 1] != 3) {
                        score += 100;
                    }
                } else {
                    score += (column - j) * 10;
                    if (j - 1 >= 0 && map[i][j - 1] != 4 &&
                        map[i][j - 1] != 1 && map[i][j - 1] != 2 &&
                        map[i][j - 1] != 3) {
                        score += 100;
                    }
                }

                int dist = manhattanDistance(person.x, person.y, i, j);
                score -= dist * 2;

                int alignDist =
                    distanceToNearestGoalRow(map, side, row, column, i);
                if (alignDist == std::numeric_limits<int>::max()) {
                    score -= 5000;
                } else {
                    score -= alignDist * 120;
                }

                if (goalDirectionBlocked(map, side, boxPos, row, column) &&
                    verticallyTrapped(map, boxPos, row, column)) {
                    score -= 6000;
                }

                if (opponentAnchorsGoalDirection(map, side, boxPos, row,
                                                  column)) {
                    score -= 6500;
                } else if (opponentBlocksGoalDirection(map, side, boxPos, row,
                                                        column)) {
                    score -= 1800;
                }

                if (score > bestScore) {
                    bestScore = score;
                    best = boxPos;
                }
            }
        }
    }

    return best;
}

// 评估移动的得分
int evaluateMove(const std::vector<std::vector<int>>& map, int side,
                 Position person, int direction, int row, int column) {
  int score = 0;
  int nx = person.x + dx[direction];
  int ny = person.y + dy[direction];

  if (nx < 0 || nx >= row || ny < 0 || ny >= column) return -10000;
  if (map[nx][ny] == 4) return -10000;

  int opponent = (side == 1) ? 2 : 1;
  if (map[nx][ny] == opponent) return -10000;
  if (map[nx][ny] == side) return -10000;

  // 目标列（箱子进入该列视为到位）
  int target_col = (side == 1) ? column - 2 : 1;

  if (map[nx][ny] == 3) {
    // 如果正在推动的箱子已经在目标列，则强烈惩罚该推动（避免把已到位箱子再去推）
    if (ny == target_col) {
      return -10000;
    }

    if (opponentOpposesPush(map, side, person, direction, row, column)) {
      return -9500;
    }

    int bx = nx + dx[direction];
    int by = ny + dy[direction];

    if (bx < 0 || bx >= row || by < 0 || by >= column) return -10000;
    if (map[bx][by] == 4) return -10000;
    if (map[bx][by] == side || map[bx][by] == opponent) return -10000;
    if (map[bx][by] == 3) return -10000;

    if ((side == 1 && by < ny) || (side == 2 && by > ny)) {
      return -8000;
    }

    Position newBox(bx, by);
    bool reachedGoalColumn =
        (side == 1 && by == column - 2) || (side == 2 && by == 1);
    if (!reachedGoalColumn &&
        goalDirectionBlocked(map, side, newBox, row, column) &&
        verticallyTrapped(map, newBox, row, column)) {
      return -9000;
    }

    if (direction == 0 || direction == 1) {
      int currentAlign = distanceToNearestGoalRow(map, side, row, column, nx);
      int nextAlign = distanceToNearestGoalRow(map, side, row, column, bx);
      if (currentAlign == std::numeric_limits<int>::max()) {
        score -= 200;
      } else if (nextAlign < currentAlign) {
        score += 600 + (currentAlign - nextAlign) * 150;
      } else if (nextAlign > currentAlign) {
        score -= (nextAlign - currentAlign) * 200;
      } else {
        score += 40;
      }
    }

    // 推入目标列奖励很高
    if (side == 1 && by == column - 2) {
      score += 10000;
    } else if (side == 2 && by == 1) {
      score += 10000;
    } else {
      if (side == 1 && by > ny) {
        score += 500 + (by - ny) * 50;
      } else if (side == 2 && by < ny) {
        score += 500 + (ny - by) * 50;
      } else {
        score += 5;
      }
    }
  } else {
    Position targetBox = findBestBox(map, side, person, row, column);

    if (targetBox.x != -1) {
      int distBefore =
          manhattanDistance(person.x, person.y, targetBox.x, targetBox.y);
      int distAfter = manhattanDistance(nx, ny, targetBox.x, targetBox.y);

      bool wrongBefore = (side == 1 && person.y >= targetBox.y) ||
                         (side == 2 && person.y <= targetBox.y);
      bool wrongAfter =
          (side == 1 && ny >= targetBox.y) || (side == 2 && ny <= targetBox.y);

      if (distAfter < distBefore) {
        score += 200 - distAfter * 10;
      } else if (distAfter == distBefore) {
        if (side == 1 && ny > person.y) {
          score += 20;
        } else if (side == 2 && ny < person.y) {
          score += 20;
        } else {
          score += 5;
        }
      } else {
        score += 2;
      }

      if (wrongAfter) {
        score -= 500;
      } else if (wrongBefore && !wrongAfter) {
        score += 250;
      }

            if (distAfter >= distBefore) {
                if (opponentAnchorsGoalDirection(map, side, targetBox, row, column)) {
                    score -= 600;
                } else if (opponentBlocksGoalDirection(map, side, targetBox, row,
                                                                                             column)) {
                    score -= 200;
                }
            }
    } else {
      if (side == 1 && ny > person.y) {
        score += 10;
      } else if (side == 2 && ny < person.y) {
        score += 10;
      } else {
        score += 1;
      }
    }

    // 目的地列占位惩罚，避免己方人员挡在箱子前方
    if ((side == 1 && ny >= target_col) || (side == 2 && ny <= target_col)) {
      score -= 2000;
    }

    if (opponentMayPushInto(map, side, Position(nx, ny), row, column)) {
      score -= 9000;
    }
  }

  return score;
}

// 检查人员是否能有效移动（不是被困住）
bool canPersonMove(const std::vector<std::vector<int>> &map, Position person,
                   int row, int column) {
    int validMoves = 0;
    for (int dir = 0; dir < 4; dir++) {
        int nx = person.x + dx[dir];
        int ny = person.y + dy[dir];

        if (nx >= 0 && nx < row && ny >= 0 && ny < column) {
            if (map[nx][ny] == 0 || map[nx][ny] == 3) {
                validMoves++;
            }
        }
    }
    return validMoves > 0;
}

// 检查人员在目标方向上是否有直接障碍
bool hasDirectObstacle(const std::vector<std::vector<int>> &map, int side,
                       Position person, Position target, int row, int column) {
    int dx_to_target = target.x - person.x;
    int dy_to_target = target.y - person.y;

    int blockedDirections = 0;
    int totalDirections = 0;

    if (side == 1 && person.y < target.y) {
        totalDirections++;
        int ny = person.y + 1;
        if (ny >= column || map[person.x][ny] == 4 ||
            map[person.x][ny] == 1 || map[person.x][ny] == 2) {
            blockedDirections++;
        }
    } else if (side == 2 && person.y > target.y) {
        totalDirections++;
        int ny = person.y - 1;
        if (ny < 0 || map[person.x][ny] == 4 ||
            map[person.x][ny] == 1 || map[person.x][ny] == 2) {
            blockedDirections++;
        }
    }

    if (dx_to_target != 0) {
        totalDirections++;
        int dir = (dx_to_target > 0) ? 1 : 0;
        int nx = person.x + dx[dir];
        if (nx < 0 || nx >= row || map[nx][person.y] == 4 ||
            map[nx][person.y] == 1 || map[nx][person.y] == 2) {
            blockedDirections++;
        }
    }

    return totalDirections > 0 && blockedDirections == totalDirections;
}

// 计算人员到最优箱子的实际可达距离（考虑障碍）
int calculatePersonBoxScore(const std::vector<std::vector<int>> &map, int side,
                            Position person, int row, int column) {
    Position targetBox = findBestBox(map, side, person, row, column);

    if (targetBox.x == -1) return -1000;

    int dist = manhattanDistance(person.x, person.y, targetBox.x, targetBox.y);
    int score = 1000 - dist * 10;

    int alignDist =
        distanceToNearestGoalRow(map, side, row, column, targetBox.x);
    if (alignDist == std::numeric_limits<int>::max()) {
        score -= 2000;
    } else {
        score -= alignDist * 80;
    }

    if (!canPersonMove(map, person, row, column)) {
        score -= 800;
    }

    if (hasDirectObstacle(map, side, person, targetBox, row, column)) {
        score -= 600;
    }

    if ((side == 1 && person.y >= targetBox.y) ||
        (side == 2 && person.y <= targetBox.y)) {
        score -= 500;
    }

    if (opponentAnchorsGoalDirection(map, side, targetBox, row, column)) {
        score -= 4000;
    } else if (opponentBlocksGoalDirection(map, side, targetBox, row,
                                           column)) {
        score -= 1200;
    }

    for (int dir = 0; dir < 4; dir++) {
        int nx = person.x + dx[dir];
        int ny = person.y + dy[dir];

        if (nx >= 0 && nx < row && ny >= 0 && ny < column && map[nx][ny] == 3) {
            int bx = nx + dx[dir];
            int by = ny + dy[dir];

            if (bx >= 0 && bx < row && by >= 0 && by < column &&
                map[bx][by] != 4 && map[bx][by] != 1 && map[bx][by] != 2 &&
                map[bx][by] != 3) {
                if ((side == 1 && by > ny) || (side == 2 && by < ny)) {
                    score += 3000;
                }
            }
        }
    }

    return score;
}

// 执行移动并更新地图
bool executeMove(std::vector<std::vector<int>> &map, Position person, int direction,
                 int side, int row, int column) {
    int nx = person.x + dx[direction];
    int ny = person.y + dy[direction];

    if (nx < 0 || nx >= row || ny < 0 || ny >= column) return false;
    if (map[nx][ny] == 4) return false;

    int opponent = (side == 1) ? 2 : 1;
    if (map[nx][ny] == side || map[nx][ny] == opponent) return false;

    if (map[nx][ny] == 3) {
        int bx = nx + dx[direction];
        int by = ny + dy[direction];

        if (bx < 0 || bx >= row || by < 0 || by >= column) return false;
        if (map[bx][by] != 0) return false;

        map[bx][by] = 3;
        map[nx][ny] = side;
        map[person.x][person.y] = 0;
    } else {
        map[nx][ny] = side;
        map[person.x][person.y] = 0;
    }

    return true;
}

int evaluateState(const std::vector<std::vector<int>> &map, int side,
                  int row, int column) {
    std::vector<Position> boxes;
    std::vector<Position> persons;
    boxes.reserve(row - 3);
    persons.reserve(6);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (map[i][j] == 3) {
                boxes.emplace_back(i, j);
            } else if (map[i][j] == side) {
                persons.emplace_back(i, j);
            }
        }
    }

    int score = -calculateHeuristic(map, side, boxes, row, column);
    int mobility = 0;

    for (const auto &person : persons) {
        mobility += calculatePersonBoxScore(map, side, person, row, column);
    }

    if (mobility == 0) {
        return score;
    }

    return score + mobility / 3;
}

int depthSearch(std::vector<std::vector<int>> &map, int side, int row,
                int column, int depth,
                const std::chrono::steady_clock::time_point &start,
                std::unordered_set<size_t> &visited) {
    if (depth == 0 || timeExceeded(start)) {
        return evaluateState(map, side, row, column);
    }

    GameState state = getCurrentState(map, side, row, column);
    auto insertion = visited.insert(state.hash());
    if (!insertion.second) {
        return evaluateState(map, side, row, column);
    }

    int best = NEG_INF;
    bool found = false;

    std::vector<Position> persons;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (map[i][j] == side) {
                persons.emplace_back(i, j);
            }
        }
    }

    for (const auto &person : persons) {
        if (timeExceeded(start)) break;
        for (int dir = 0; dir < 4; ++dir) {
            if (timeExceeded(start)) break;
            std::vector<std::vector<int>> next = map;
            if (!executeMove(next, person, dir, side, row, column)) {
                continue;
            }
            int immediate = evaluateMove(map, side, person, dir, row, column);
            int future = depthSearch(next, side, row, column, depth - 1, start, visited);
            int total = immediate + future;
            if (total > best) {
                best = total;
            }
            found = true;
        }
    }

    visited.erase(insertion.first);

    if (!found) {
        return evaluateState(map, side, row, column);
    }
    return best;
}

// 寻找最佳移动（主入口，结合快速启发式和IDA*搜索）
Move findBestMove(const std::string &gameId,
                  const std::vector<std::vector<int>> &map, int side, int row,
                  int column) {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<Position> myPersons;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (map[i][j] == side) {
                myPersons.emplace_back(i, j);
            }
        }
    }

    if (myPersons.empty()) {
        return Move(Position(), 0, 0);
    }

    const std::string memoryKey = gameId + "#" + std::to_string(side);
    auto &lastDirections = g_movementMemory[memoryKey].lastDirection;

    std::unordered_set<int> activeKeys;
    activeKeys.reserve(myPersons.size());
    for (const auto &person : myPersons) {
        activeKeys.insert(encodePositionKey(person));
    }

    for (auto it = lastDirections.begin(); it != lastDirections.end();) {
        if (activeKeys.find(it->first) == activeKeys.end()) {
            it = lastDirections.erase(it);
        } else {
            ++it;
        }
    }

    auto oscillationPenalty = [&](const Position &person, int direction) {
        auto it = lastDirections.find(encodePositionKey(person));
        if (it != lastDirections.end() && ((it->second ^ 1) == direction)) {
            return OSCILLATION_PENALTY;
        }
        return 0;
    };

    auto recordChosenMove = [&](const Position &person, int direction) {
        std::vector<std::vector<int>> simulation = map;
        if (!executeMove(simulation, person, direction, side, row, column)) {
            return;
        }
        lastDirections.erase(encodePositionKey(person));
        Position nextPos(person.x + dx[direction], person.y + dy[direction]);
        lastDirections[encodePositionKey(nextPos)] = direction;
    };

    const int target_col = (side == 1) ? column - 2 : 1;
    std::vector<Position> candidateBoxes;
    candidateBoxes.reserve(row * column);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (map[i][j] == 3) {
                if ((side == 1 && j >= target_col) ||
                    (side == 2 && j <= target_col)) {
                    continue;
                }
                candidateBoxes.emplace_back(i, j);
            }
        }
    }

    auto nearestBoxDistance = [&](const Position &person) {
        int best = std::numeric_limits<int>::max();
        for (const auto &box : candidateBoxes) {
            best = std::min(best,
                            manhattanDistance(person.x, person.y,
                                              box.x, box.y));
        }
        return (best == std::numeric_limits<int>::max()) ? 0 : best;
    };

    std::stable_sort(myPersons.begin(), myPersons.end(),
                     [&](const Position &a, const Position &b) {
                         return nearestBoxDistance(a) <
                                nearestBoxDistance(b);
                     });

    for (const auto &person : myPersons) {
        for (int dir = 0; dir < 4; ++dir) {
            int nx = person.x + dx[dir];
            int ny = person.y + dy[dir];
            if (nx < 0 || nx >= row || ny < 0 || ny >= column) {
                continue;
            }
            if (map[nx][ny] != 3) {
                continue;
            }
            int bx = nx + dx[dir];
            int by = ny + dy[dir];
            if (bx < 0 || bx >= row || by < 0 || by >= column) {
                continue;
            }
            if (map[bx][by] != 0) {
                continue;
            }
            if ((side == 1 && by == target_col) ||
                (side == 2 && by == target_col)) {
                recordChosenMove(person, dir);
                return Move(person, dir, 20000);
            }
        }
    }

    Move fallback = Move(myPersons.front(), 0, NEG_INF);
    int fallbackScore = NEG_INF;
    bool timeout = false;

    for (const auto &person : myPersons) {
        if (timeExceeded(start_time)) {
            timeout = true;
            break;
        }
        for (int dir = 0; dir < 4; ++dir) {
            if (timeExceeded(start_time)) {
                timeout = true;
                break;
            }
            std::vector<std::vector<int>> scratch = map;
            if (!executeMove(scratch, person, dir, side, row, column)) {
                continue;
            }
            int score = evaluateMove(map, side, person, dir, row, column);
            score -= oscillationPenalty(person, dir);
            if (score > fallbackScore) {
                fallbackScore = score;
                fallback = Move(person, dir, score);
            }
        }
        if (timeout) break;
    }

    if (fallbackScore == NEG_INF) {
        return Move(myPersons.front(), 0, 0);
    }

    Move bestMove = fallback;
    int bestScore = fallbackScore;

    if (timeout) {
        recordChosenMove(bestMove.person, bestMove.direction);
        return bestMove;
    }

    constexpr int SEARCH_DEPTH = 4;

    for (const auto &person : myPersons) {
        if (timeExceeded(start_time)) break;
        for (int dir = 0; dir < 4; ++dir) {
            if (timeExceeded(start_time)) break;
            std::vector<std::vector<int>> next = map;
            if (!executeMove(next, person, dir, side, row, column)) {
                continue;
            }
            int immediate = evaluateMove(map, side, person, dir, row, column);
            immediate -= oscillationPenalty(person, dir);
            std::unordered_set<size_t> visited;
            int future = depthSearch(next, side, row, column, SEARCH_DEPTH - 1,
                                     start_time, visited);
            int total = immediate + future;
            if (total > bestScore) {
                bestScore = total;
                bestMove = Move(person, dir, total);
            }
        }
    }

    recordChosenMove(bestMove.person, bestMove.direction);
    return bestMove;
}

}  // namespace

int main() {
    drogon::app().addListener("0.0.0.0", 8081);

    drogon::app().registerHandler(
        "/api/run",
        [](const drogon::HttpRequestPtr &req,
           std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
            // LOG_INFO << "Received /api/run request: " << req->getBody();
            const auto json = req->getJsonObject();
            if (!json) {
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                resp->setBody(R"({"error":"invalid json payload"})");
                callback(resp);
                return;
            }

            try {
                std::string uid = (*json)["uid"].asString();
                int side = (*json)["side"].asInt();
                int row = (*json)["row"].asInt();
                int column = (*json)["column"].asInt();

                std::vector<std::vector<int>> map(row,
                                                   std::vector<int>(column));
                const auto &mapArray = (*json)["map"];
                for (int i = 0; i < row; i++) {
                    for (int j = 0; j < column; j++) {
                        map[i][j] = mapArray[i][j].asInt();
                    }
                }

                // LOG_INFO << "Game " << uid << ", Side: " << side
                //          << ", Map: " << row << "x" << column;

                Move bestMove = findBestMove(uid, map, side, row, column);

                Json::Value responseJson;
                responseJson["direction"] = bestMove.direction;
                responseJson["position"] = Json::Value(Json::arrayValue);
                responseJson["position"].append(bestMove.person.x);
                responseJson["position"].append(bestMove.person.y);

                auto resp = drogon::HttpResponse::newHttpJsonResponse(
                    responseJson);
                resp->setStatusCode(drogon::k200OK);
                callback(resp);

            } catch (const std::exception &e) {
                LOG_ERROR << "Error processing request: " << e.what();
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k500InternalServerError);
                resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                resp->setBody(R"({"error":"internal server error"})");
                callback(resp);
            }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/run",
        [](const drogon::HttpRequestPtr &,
           std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k204NoContent);
            callback(resp);
        },
        {drogon::Options});

    drogon::app().registerPostHandlingAdvice(
        [](const drogon::HttpRequestPtr &,
           const drogon::HttpResponsePtr &resp) { addCorsHeaders(resp); });

    drogon::app().run();
    return 0;
}
