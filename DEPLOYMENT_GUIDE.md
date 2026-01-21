# 🚀 傻瓜式部署指南: JP-DocRAG-Eval

这份指南专门为你准备。我们使用 **Docker** 容器化部署，这能屏蔽所有环境问题。
同时，鉴于项目中的数据文件 (artifacts, data) 没有纳入 Git 管理，我们采用 **"打包上传"** 的方式。

---

## 第一步：在本地打包 📦

既然你现在的本地环境是好的，我们直接把所有文件打个包。
我在项目根目录写好了一个脚本 `package_upload.sh`。

1.  在你的 Mac 终端里运行：
    ```bash
    ./package_upload.sh
    ```
2.  你会发现多了一个 **`project_deploy.zip`** 文件。
3.  **把这个 ZIP 文件传到你的 Linux 服务器上。**
    *   你可以用 `scp` 命令，或者 FileZilla 这种可视化工具。

---

## 第二步：在服务器上解压运行 🏃‍♂️

登录你的 Linux 服务器，进入你上传文件的目录。

**1. 解压文件**
```bash
# 如果没有 unzip 命令，先装一下: sudo apt install unzip (Ubuntu) 或 sudo yum install unzip (CentOS)
unzip project_deploy.zip -d jprag-demo
cd jprag-demo
```

**2. 安装 Docker (如果还没有的话)**
如果不确定有没有，输入 `docker -v` 看看。如果没有：
```bash
# Ubuntu 一键安装
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

**3. 启动项目**
这是最关键的一步，只需要这一行命令：
```bash
# -d 表示后台运行
sudo docker compose up -d --build
```
*(注: 旧版 Docker 可能需要用 `docker-compose up -d --build`)*

**4. 检查是否成功**
```bash
sudo docker compose ps
```
如果你看到状态是 `Up`，那就说明跑起来了！

---

## 第三步：如何访问网页？ 🌐

因为这是部署在远程服务器上，你不能直接访问 `localhost`。有两种方法：

### 方法 A: SSH 隧道 (最推荐，不需要搞服务器防火墙) 👍
这是最安全、最简单的办法。它会把你本地电脑的 8501 端口直连到服务器的 8501 端口。

在你的 **本地 Mac 终端** (不是服务器终端) 运行：
```bash
# 把 username@remote_ip 换成你登录服务器的账号IP
ssh -L 8501:localhost:8501 username@remote_ip
```
登录进去后挂着别关。
然后打开你本地浏览器，访问 **[http://localhost:8501](http://localhost:8501)**
🎉 你应该就能看到网页了！

### 方法 B: 直接 IP 访问 (需要开防火墙)
如果你的服务器有公网 IP，且你会在安全组里放行端口。
1.  在阿里云/AWS/腾讯云控制台，“安全组”设置里，允许 **TCP 8501** 端口的入站流量。
2.  直接访问 `http://<服务器IP>:8501`。

---

## 常见问题

*   **Q: 报错 API Key 找不到？**
    A: 检查一下解压出来的目录里有没有 `.env` 文件 (`ls -a`)。如果没有，手动建一个：`nano .env` 然后把 Key 贴进去。
*   **Q: 修改了代码怎么办？**
    A: 重新打包上传覆盖，或者在服务器上直接改文件，然后运行 `sudo docker compose restart`。

---

## 💡 运维小贴士 (Pro Tips)

**1. 查看实时日志**
如果有报错或想看后台输出，运行：
```bash
sudo docker compose logs -f --tail=200
```

**2. 免 sudo 运行 Docker**
为了方便，可以把当前用户加入 docker 组（需要重新登录生效）：
```bash
sudo usermod -aG docker $USER
# 注销并重新登录
logout
```
之后就可以直接运行 `docker compose up` 而不用 `sudo` 了。

**3. 数据持久化机制**
我们在 `docker-compose.yml` 中配置了 **Volumes**：
*   `./data:/app/data`
*   `./artifacts:/app/artifacts`

**好处**：
*   **镜像小**：`.dockerignore` 忽略了这两个大文件夹，构建镜像非常快。
*   **数据保全**：即使你删除了容器 (`docker compose down`)，你的索引和数据依然保存在服务器的文件夹里，不会丢失。下次启动直接挂载使用。
