# Git

## 目录

- [基础操作](#基础操作)
- [分支管理](#分支管理)
- [远程操作](#远程操作)
- [撤销与回退](#撤销与回退)
- [更新代码](#更新代码)
- [常用场景](#常用场景)
  - [获取代码](#获取代码)
  - [删除本地分支](#删除本地分支)
  - [切换远程分支](#切换远程分支)
  - [更新代码](#更新代码)
  - [代码提交到了错误分支](#代码提交到了错误分支)
  - [不commit本地代码就合并远程分支](#不commit本地代码就合并远程分支)

[git连接远程仓库](https://blog.csdn.net/x12301/article/details/120674441?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522409859fe95cb3b6b2fbcfb2838dbd6f5%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D\&request_id=409859fe95cb3b6b2fbcfb2838dbd6f5\&biz_id=0\&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-120674441-null-null.142^v102^pc_search_result_base2\&utm_term=git%E8%BF%9E%E6%8E%A5%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93\&spm=1018.2226.3001.4187 "git连接远程仓库")

# 基础操作

| 命令                      | 功能说明             |
| ----------------------- | ---------------- |
| \`git init\`            | 初始化新仓库（创建.git目录） |
| \`git clone \<url>\`    | 克隆远程仓库到本地        |
| \`git add \<file>\`     | 添加文件到暂存区（准备提交）   |
| \`git commit -m "msg"\` | 提交暂存区内容到本地仓库     |
| \`git status\`          | 查看工作区和暂存区状态      |
| \`git log\`             | 查看提交历史记录         |
| \`git diff\`            | 查看工作区与暂存区的差异     |

***

# 分支管理

| 命令                                | 功能说明                 |
| --------------------------------- | -------------------- |
| \`git branch\`                    | 列出所有\*\*本地\*\*分支     |
| \`git branch -a\`                 | 列出所有分支（本地+远程）        |
| \`git branch \<name>\`            | 创建新分支（不切换）           |
| \`git checkout \<branch>\`        | 切换到指定分支              |
| \`git checkout -b \<new-branch>\` | \*\*创建并切换\*\*到新分支    |
| \`git switch \<branch>\`          | (Git 2.23+) 更安全的分支切换 |
| \`git merge \<branch>\`           | 合并指定分支到\*\*当前分支\*\*  |
| \`git branch -d \<branch>\`       | \*\*删除本地分支\*\*（已合并）  |
| \`git branch -D \<branch>\`       | 强制删除本地分支（未合并）        |

***

# 远程操作

| 命令                               | 功能说明                   |
| -------------------------------- | ---------------------- |
| \`git fetch\`                    | 从远程仓库下载变更（不合并）         |
| \`git pull\`                     | 拉取远程变更并\*\*合并到当前分支\*\* |
| \`git push\`                     | 推送本地提交到远程仓库            |
| \`git push -u origin \<branch>\` | \*\*首次推送分支\*\*并建立跟踪    |
| \`git remote -v\`                | 查看远程仓库地址               |
| \`git remote show origin\`       | 查看远程仓库详细信息             |

***

# 撤销与回退

| 命令                               | 功能说明                   |
| -------------------------------- | ---------------------- |
| \`git restore \<file>\`          | 撤销工作区的修改（未add）         |
| \`git restore --staged \<file>\` | 将文件移出暂存区（已add未commit）  |
| \`git reset --soft HEAD\~1\`     | 撤销上次提交（保留修改在暂存区）       |
| \`git reset --hard HEAD\~1\`     | \*\*彻底回退\*\*到上次提交（慎用！） |
| \`git revert \<commit-id>\`      | 创建新提交来撤销指定提交（安全回退）     |

# 更新代码

从开发主分支上更新代码并合并到自己的分支

```markdown 
# 加入master是开发主分支
git checkout master

git pull

# 切换到自己的分支
git checkout fuhuabin

# 从master合并过来
git merge --no-ff master

# 合并成功后提交到远程
git push
```


# 常用场景

```markdown 
//撤销未提交的更改
git checkout -- .



```


## 获取代码

获取远程代码并创建本地分支来关联远程origin/dev分支，git clone默认关联的是origin/main分支，要自行切换

```markdown 
//获取远程仓库代码
git clone 远程仓库地址

//创建本地分支来关联远程origin/per/guoruilei/dev分支
git checkout -b per/guoruilei/dev origin/per/guoruilei/dev
```


## 删除本地分支

```markdown 
//首先查看所有本地分支：
git branch

//确保你不在要删除的分支上（切换到其他分支）：
git checkout 主分支名  # 如 git checkout main

//执行删除命令：
git branch -d 要删除的分支名

```


## 切换远程分支

- 我在本地是git checkout -b my-branch-name origin/dev关联远程分支，然后使用git push origin my-branch-name推送分支到远程，那么我应该如将修改后的代码推送到我的远程分支而不是origin/dev分支

```markdown 
 //创建远程分支并并联
 git push -u origin per/guoruilei/dev
 
 //确认当前分支的追踪关系
 git branch -vv
 
 //修正分支的追踪关系
 //如果当前分支关联的是 origin/dev，需将其改为关联远程的同名分支
 git branch --set-upstream-to=origin/per/guoruilei2/dev per/guoruilei/dev
 
 //确保处于正确的远程分支
 git checkout my-branch-name
```


## 更新代码

- 将远程分支origin/dev的修改合并到本地（当前关联origin/dev分支）

```markdown 
//获取远程关联分支更新
git fetch

//合并远程更改到本地
git merge origin/dev

```


- 将远程分支origin/dev的修改合并到本地（当前不是关联origin/dev分支）

```javascript 
//获取远程所有分支的更新
git fetch origin

//合并远程更改到本地
git merge origin/dev

```


## 代码提交到了错误分支

本来想要提交本地per/guoruilei/dev分支的代码到origin/per/guoruilei/dev分支上的，但是不小心在本地的per/guoruilei/demo分支上修改了代码，并且提交到了origin/per/guoruilei/demo分支上

```markdown 
//解除per/guoruilei/demo分支的关联
git checkout per/guoruilei/demo
git branch --unset-upstream

//将本地per/guoruilei/dev分支关联到origin/per/guoruilei/demo分支
git checkout per/guoruilei/dev
git branch --set-upstream-to=origin/per/guoruilei/demo per/guoruilei/dev

//获取origin/per/guoruilei/demo分支的最新代码
git fetch
git merge origin/per/guoruilei/demo

//解除per/guoruilei/dev分支到origin/per/guoruilei/demo分支的绑定
git branch --unset-upstream

//恢复原来的分支绑定
git branch --set-upstream-to=origin/per/guoruilei/dev per/guoruilei/dev
git checkout per/guoruilei/demo
git branch --set-upstream-to=origin/per/guoruilei/demo per/guoruilei/demo


```


## 不commit本地代码就合并远程分支

可以使用git stash暂存本地的修改，再合并远程分支，合并完成后就可以将暂存的修改恢复回来

1. 保存更改
   ```markdown 
   //将本地的修改储藏起来
   git stash（git stash是git stash push的简写）

   //给stash添加描述性标识
   git stash -m "描述性信息"

   //git stash不会包括未跟踪的文件（新创建但从未添加到版本控制的文件）,包含这些修改要加上-u参数
   git stash -u


   ```

2. 拉取远程代码
   ```bash 
   git fetch 
   git merge origin/dev
   ```

3. 恢复本地修改
   ```markdown 
   //恢复最新一次储藏，并​​将这次储藏从储藏列表中删除​​（有冲突时要手动删除）
   git stash pop

   //恢复指定储藏
   //查看储藏列表
   git stash list   
   //恢复索引为1的储藏   
   git stash apply 1
   ```

4. 处理可能冲突

   如果远程的修改与您储藏的修改恰好发生在同一文件的相同位置，可能会产生​**​冲突​**​。您需要手动解决这些冲突
5. 丢弃储藏
   | 你的目标          | 核心命令                        | 关键解释                                             |
   | ------------- | --------------------------- | ------------------------------------------------ |
   | **丢弃单个指定储藏**​ | \`git stash drop \<index>\` | 精准删除特定储藏（例如 git stash drop 1），删除后\*\*无法常规恢复\*\*。 |
   | **丢弃最新的储藏**​  | \`git stash drop\`          | 这是最快捷的方式，默认删除栈顶的 \`stash@{0}\`。                  |
   | **清空整个储藏栈**​  | \`git stash clear\`         | \*\*一次性永久删除所有储藏\*\*，请务必谨慎使用。                     |
