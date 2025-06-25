# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：



代码：

    class Solution:
        def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
            def helper(left, right):
                if left > right:
                    return None

                mid = (left + right) // 2
                node = TreeNode(nums[mid])
                node.left = helper(left, mid - 1)
                node.right = helper(mid + 1, right)
                return node

            return helper(0, len(nums) - 1)



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-01](D:\大一下课程\数据结构与算法B\作业)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：



代码：

    class TreeNode:
        def __init__(self, value):
            self.value = value
            self.children = []


    def traverse_print(root, nodes):
        if root.children == []:
            print(root.value)
            return
        pac = {root.value: root}
        for child in root.children:
            pac[child] = nodes[child]
        for value in sorted(pac.keys()):
            if value in root.children:
                traverse_print(pac[value], nodes)
            else:
                print(root.value)


    n = int(input())
    nodes = {}
    children_list = []
    for i in range(n):
        info = list(map(int, input().split()))
        nodes[info[0]] = TreeNode(info[0])
        for child_value in info[1:]:
            nodes[info[0]].children.append(child_value)
            children_list.append(child_value)
    root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
    traverse_print(root, nodes)



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-02](D:\大一下课程\数据结构与算法B\作业)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：



代码：

    class Solution:
        def sumNumbers(self, root: Optional[TreeNode]) -> int:
            def dfs(node, current_sum):
                if not node:
                    return 0
                current_sum = current_sum * 10 + node.val
                if not node.left and not node.right:
                    return current_sum
                return dfs(node.left, current_sum) + dfs(node.right, current_sum)

            return dfs(root, 0)



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-03](D:\大一下课程\数据结构与算法B\作业)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：



代码：

    def build_postorder(preorder, inorder):
        if not preorder:
            return ''
        root = preorder[0]
        root_index = inorder.index(root)
        # 左子树
        left = build_postorder(preorder[1:1 + root_index], inorder[:root_index])
        # 右子树
        right = build_postorder(preorder[1 + root_index:], inorder[root_index + 1:])
        return left + right + root

    try:
        while True:
            preorder = input().strip()
            inorder = input().strip()
            print(build_postorder(preorder, inorder))
    except EOFError:
        pass



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-04](D:\大一下课程\数据结构与算法B\作业)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：



代码：

    class Node:
        def __init__(self, value):
            self.value = value
            self.children = []
    def parse_tree(s):
        stack = []
        root = None
        current = None
        i = 0
        while i < len(s):
            if s[i].isalpha():
                node = Node(s[i])
                if not root:
                    root = node
                if stack:
                    stack[-1].children.append(node)
                current = node
            elif s[i] == '(':
                stack.append(current)
            elif s[i] == ')':
                stack.pop()
            i += 1
        return root

    def preorder(node):
        if not node:
            return ''
        result = node.value
        for child in node.children:
            result += preorder(child)
        return result
    def postorder(node):
        if not node:
            return ''
        result = ''
        for child in node.children:
            result += postorder(child)
        result += node.value
        return result

    s = input().strip()
    root = parse_tree(s)
    print(preorder(root))
    print(postorder(root))



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-05](D:\大一下课程\数据结构与算法B\作业)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：



代码：

    from sortedcontainers import SortedList

    class Solution:
        def minimumPairRemoval(self, nums: List[int]) -> int:
            n = len(nums)
            sl = SortedList()
            idx = SortedList(range(n))
        
            need = 0
            for i in range(n - 1):
                if nums[i] > nums[i + 1]:
                    need += 1
                sl.add((nums[i] + nums[i + 1], i))
        
            ans = 0
            while need > 0:
                s, i = sl.pop(0)
                k = idx.bisect_left(i)
            
                if k > 0:
                    pre = idx[k - 1]
                    sl.remove((nums[pre] + nums[i], pre))
                    sl.add((nums[pre] + s, pre))
                    if nums[pre] > nums[i]:
                        need -= 1
                    if nums[pre] > s:
                        need += 1
            
                nxt1 = idx[k + 1]
                if nums[i] > nums[nxt1]:
                    need -= 1

                if k + 2 < len(idx):
                    nxt2 = idx[k + 2]
                    sl.remove((nums[nxt1] + nums[nxt2], nxt1))
                    sl.add((s + nums[nxt2], i))
                    if nums[nxt1] > nums[nxt2]:
                        need -= 1
                    if s > nums[nxt2]:
                        need += 1
            
                nums[i] = s
                idx.remove(nxt1)
                ans += 1
            return ans



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![08-06](D:\大一下课程\数据结构与算法B\作业)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次的作业居然全都是树，简直就不是我这小白能够解出来的。每次看懂了一个树代码，在解下一题时又觉得自己学废了。
再次真心祈祷考试题目能完美避开树相关的题，也再次告诉自己考试遇到树就跳过吧。又是讨厌树的一天，呵呵🙂。








