# 从服务器上传 EdgeRCA 到 GitHub

在服务器终端执行 Git 命令即可上传，不需要先下载到自己的电脑，也不需要管理员或 GPU 资源。本指南是首次上传到一个空仓库的步骤；当前项目尚未初始化 Git，本次准备没有执行提交或上传。

## 先准备仓库地址和提交身份

需要确定：GitHub 仓库属于哪个账号或组织、仓库名、公开还是私有，以及提交显示名和邮箱。邮箱可以使用 GitHub 账户设置中的 noreply 地址。

在 GitHub 创建仓库，例如 `EdgeRCA`。建议先选择 Private（私有）。首次使用下面的步骤时，创建空仓库，不勾选添加 README、.gitignore 或许可证，项目里已有这些内容或需要单独决定许可证。

创建后复制 SSH 地址，形式为：

```text
git@github.com:你的用户名/EdgeRCA.git
```

仓库地址可以提供给协助上传的人；账号密码、访问 Token、SSH 私钥不需要发给他人。

## 配置服务器的 GitHub 认证

如果这台服务器已经能用 SSH 登录你的 GitHub 账号，可以复用现有配置，跳过生成密钥。

```bash
ssh -T git@github.com
```

成功信息中会显示你的 GitHub 用户名，以及不提供 shell 的说明。此命令即使认证成功，退出码也可能为 1，以认证成功文字为准。

如果尚未配置，先确认专用密钥文件是否存在：

```bash
ls ~/.ssh/edgerca_github*
```

没有这些文件时，生成一对专用密钥；已有时复用，不覆盖：

```bash
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -C "你的GitHub邮箱" -f ~/.ssh/edgerca_github
cat ~/.ssh/edgerca_github.pub
```

按提示设置密钥口令。将最后一条命令显示的整行公钥复制到 GitHub 的 Settings → SSH and GPG keys → New SSH key，密钥类型选择 Authentication Key。`.pub` 是可提交给 GitHub 的公钥；没有 `.pub` 后缀的文件是私钥，保留在服务器上。

测试这把专用密钥：

```bash
ssh -i ~/.ssh/edgerca_github -o IdentitiesOnly=yes -T git@github.com
```

首次连接按 GitHub 官方说明核对主机指纹，成功后应显示你自己的 GitHub 用户名。若认证失败或服务器无法访问 GitHub，先解决连接问题，再进行上传。

## 初始化本地代码仓库

下面的路径是这台服务器上已核对的项目根目录。将显示名和邮箱替换成自己的信息；这两项是提交署名，不是登录密码。配置仅作用于本项目。

```bash
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
git init -b main
git config user.name "你的提交显示名"
git config user.email "你的GitHub邮箱或noreply邮箱"
```

如果刚才使用了专用密钥，再为本项目指定它；复用已有可用 SSH 配置时无需执行：

```bash
git config core.sshCommand 'ssh -i ~/.ssh/edgerca_github -o IdentitiesOnly=yes'
```

## 检查并提交代码

本项目的 `.gitignore` 排除了 `outputs/` 下的生成结果、模型目录、Python 缓存、环境文件，以及 `web/trace/data/` 和自动生成的 `web/trace/EdgeRCA_trace.html`。被忽略的文件仍保存在本地，不会删除。

首轮明确选择源码及文档目录，先预览，再加入暂存区：

```bash
git add --dry-run -- .gitignore AGENTS.md README.md edge_rca tests tools slurm docs web
git add -- .gitignore AGENTS.md README.md edge_rca tests tools slurm docs web
git diff --cached --stat
git diff --cached --name-only
```

确认列表是准备上传的代码和文档，且没有实验数据库、模型权重或凭据后，创建首次提交：

```bash
git commit -m "init: add EdgeRCA offline analysis code"
```

`git add` 只是选择本次提交的文件，`git commit` 只是生成本地版本；它们还没有把代码传到 GitHub。

## 关联 GitHub 并上传

将下面的地址换成自己新建的空仓库地址。首次没有 origin 时才执行 remote add；以后可用 `git remote -v` 查看已配置地址。

```bash
git remote add origin git@github.com:你的用户名/EdgeRCA.git
git push -u origin main
```

最后一条 `git push` 才真正上传。完成后刷新 GitHub 仓库页面，应能看到 `edge_rca/`、`tests/`、`tools/`、`slurm/`、`docs/` 和前端源码。

如果提示远端已经存在提交，不要直接强制推送；先核对是否建错仓库，或远端是否已经添加了 README 等文件。

## 后续更新

在同一项目目录中查看修改，选择本次要上传的文件后提交：

```bash
git status --short
git diff
git add -- 本次修改的文件或目录
git diff --cached --stat
git commit -m "fix: describe the actual change"
git push
```

上面的“本次修改的文件或目录”和提交说明是占位内容，执行前替换为真实路径和说明。

仓库中只保留前端源码和导出工具。真实案例数据与单文件离线页面需要按 `web/trace/README.md` 从本地实验目录生成；实验材料是否另行分享应单独决定。

## 官方说明

- [将本地代码添加到 GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [生成 SSH 密钥](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [测试 SSH 连接](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)
