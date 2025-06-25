# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Complied by <mark>叶靖、信管</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：



代码：

    from collections import deque

    def bfs(maze, R, C, start, end):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(start[0], start[1], 0)])
        visited = [[False] * C for _ in range(R)] # 初始化访问记录数组
        visited[start[0]][start[1]] = True
        while queue:
            x, y, steps = queue.popleft()
            if (x, y) == end:
                return steps
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < R and 0 <= ny < C and not visited[nx][ny] and maze[nx][ny] != '#':
                    visited[nx][ny] = True
                    queue.append((nx, ny, steps + 1))
        return 'oop!'

    def solve():
        T = int(input())
        for _ in range(T):
            R, C = map(int, input().split())
            maze = [input().strip() for _ in range(R)]

            start = None
            end = None
            for i in range(R):
                for j in range(C):
                    if maze[i][j] == 'S':
                        start = (i, j)
                    elif maze[i][j] == 'E':
                        end = (i, j)
            result = bfs(maze, R, C, start, end)
            print(result)

    solve()



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-01](D:\大一下课程\数据结构与算法B\作业)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：



代码：

    class Solution:
        def pathExistenceQueries(self, n, nums, maxDiff, queries):
            parent = list(range(n))
            size = [1] * n

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                rootX = find(x)
                rootY = find(y)
                if rootX != rootY:
                    if size[rootX] > size[rootY]:
                        parent[rootY] = rootX
                        size[rootX] += size[rootY]
                    else:
                        parent[rootX] = rootY
                        size[rootY] += size[rootX]

            for i in range(n - 1):
                if abs(nums[i + 1] - nums[i]) <= maxDiff:
                    union(i, i + 1)

            result = []
            for u, v in queries:
                if find(u) == find(v):
                    result.append(True)
                else:
                    result.append(False)

            return result



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-02](D:\大一下课程\数据结构与算法B\作业)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：



代码：

    import math

    org_result = list(map(float, input().split()))
    org_result.sort(reverse=True)
    num_results = len(org_result)
    target_count = math.ceil(num_results * 0.6)
    target_score = org_result[target_count - 1]

    # b / 1000000000 * target_score + 1.1 ** (b / 1000000000 * target_score) >= 85

    def adjusted_score(x, b):
        a = b / 1000000000
        return a * x + 1.1 ** (a * x)

    def solve_b(target_score):
        low, high = 1, 1000000000
        result_b = high
        while low <= high:
            mid = (low + high) // 2
            adjusted_scores = [adjusted_score(x, mid) for x in org_result]
            count = sum(1 for score in adjusted_scores if score >= 85)
            if count >= target_count:
                result_b = mid
                high = mid - 1
            else:
                low = mid + 1
        print(result_b)

    solve_b(target_score)



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-03](D:\大一下课程\数据结构与算法B\作业)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：



代码：

    import sys
    sys.setrecursionlimit(1000000)

    def has_cycle(n, graph):
        visited = [0] * n  # 0: 未访问, 1: 访问中, 2: 访问完成

        def dfs(u):
            visited[u] = 1  # 标记为访问中
            for v in graph[u]:
                if visited[v] == 0:  # 如果未访问，递归
                    if dfs(v):
                        return True
                elif visited[v] == 1:  # 如果访问中，说明有环
                    return True
            visited[u] = 2  # 标记为访问完成
            return False

        for i in range(n):
            if visited[i] == 0:
                if dfs(i):
                    return True
        return False

    # 读取输入
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)

    # 输出结果
    if has_cycle(n, graph):
        print("Yes")
    else:
        print("No")



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-04](D:\大一下课程\数据结构与算法B\作业)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：



代码：

    import heapq

    # 输入地点
    P = int(input())
    names = [input().strip() for _ in range(P)]
    name_to_index = {name: i for i, name in enumerate(names)}
    index_to_name = {i: name for i, name in enumerate(names)}

    # 建图
    Q = int(input())
    graph = [[] for _ in range(P)]
    for _ in range(Q):
        a, b, dist = input().split()
        dist = int(dist)
        u, v = name_to_index[a], name_to_index[b]
        graph[u].append((v, dist))
        graph[v].append((u, dist))  # 因为是双向路径

    # Dijkstra 算法
    def dijkstra(start):
        n = P
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in graph[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))
        return dist, prev

    # 恢复路径
    def reconstruct_path(prev, end):
        path = []
        while end != -1:
            path.append(end)
            end = prev[end]
        path.reverse()
        return path

    # 查询处理
    R = int(input())
    for _ in range(R):
        a, b = input().split()
        start, end = name_to_index[a], name_to_index[b]
        if start == end:
            print(a)
            continue
        dist, prev = dijkstra(start)
        path = reconstruct_path(prev, end)
        if not path or path[0] != start:
            print("No path")
            continue
        output = index_to_name[path[0]]
        for i in range(1, len(path)):
            u, v = path[i - 1], path[i]
            d = next(w for t, w in graph[u] if t == v)
            output += f"->({d})->{index_to_name[v]}"
        print(output)



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-05](D:\大一下课程\数据结构与算法B\作业)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：



代码：

    n = int(input())
    sr, sc = map(int, input().split())

    # 8个方向
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2),  (1, 2),  (2, -1),  (2, 1)]

    visited = [[False] * n for _ in range(n)]

    def count_onward_moves(x, y):
        """统计当前位置下一步能走的方向数"""
        count = 0
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
                count += 1
        return count

    def dfs(x, y, step):
        if step == n * n:
            return True
        next_moves = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
                c = count_onward_moves(nx, ny)
                next_moves.append((c, nx, ny))
        # 贪心：先走下一步选择更少的格子
        next_moves.sort()
        for _, nx, ny in next_moves:
            visited[nx][ny] = True
            if dfs(nx, ny, step + 1):
                return True
            visited[nx][ny] = False
        return False

    visited[sr][sc] = True
    if dfs(sr, sc, 1):
        print("success")
    else:
        print("fail")



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![B-06](D:\大一下课程\数据结构与算法B\作业)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

通过图的构建与搜索算法实践，我掌握了如何高效判断图中是否有环、求最短路径、以及解决搜索优化问题如骑士周游。
我意识到在处理大规模图或搜索问题时，盲目回溯往往超时，必须引入剪枝、排序或贪心策略（如“最少可走步数优先”）来优化算法性能。









