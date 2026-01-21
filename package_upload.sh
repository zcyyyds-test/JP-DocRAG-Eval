#!/bin/bash
# package_upload.sh
# 这是一个辅助脚本，帮你把当前项目打包成 project_deploy.zip
# 方便你通过 scp 或 SFTP 上传到服务器

echo "📦 Packaging project for deployment..."

# 删除旧的 zip
rm -f project_deploy.zip

# 压缩必要文件
# - web_demo.py: 主程序
# - manage.sh: 辅助脚本
# - requirements.txt: 依赖列表
# - Dockerfile, docker-compose.yml: 部署配置
# - src/: 源代码
# - data/: 数据文件 (包含 docs_list.csv, feedback.jsonl 等)
# - artifacts/: 索引文件 (FAISS, BM25, chunks 都在这里)
# - .env: 包含你的 API Key (请确保这里是你如果不放心可以手动删掉)

zip -r project_deploy.zip \
    web_demo.py \
    manage.sh \
    requirements.txt \
    Dockerfile \
    docker-compose.yml \
    src \
    data \
    artifacts \
    .env

echo "✅ Done! 'project_deploy.zip' created."
echo "👉 Action: Upload this file to your Linux server."
