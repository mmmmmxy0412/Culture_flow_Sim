#!/usr/bin/env python
# coding: utf-8

import json
import time
import os
from collections import defaultdict

import pandas as pd
import praw

# ====================== 代理设置（按需保留或删除） ======================
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# ====================================================================
import re


def detect_post_images(post):
    """
    检测帖子中的图片URL
    返回图片URL列表
    """
    image_urls = []

    # 1. 检查 post_hint 属性（单图帖子）
    if hasattr(post, 'post_hint'):
        if post.post_hint == 'image':
            if post.url and any(post.url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                image_urls.append(post.url)

    # 2. 检查是否是相册
    if getattr(post, 'is_gallery', False):
        try:
            if getattr(post, 'gallery_data', None):
                for item in post.gallery_data['items']:
                    media_id = item['media_id']
                    image_urls.append(f"https://i.redd.it/{media_id}.jpg")
        except Exception:
            pass

    # 3. 检查 media_metadata
    if getattr(post, 'media_metadata', None):
        for media_id, media_info in post.media_metadata.items():
            if media_info.get('e') == 'Image':  # e 表示类型，'Image' 为图片
                if 's' in media_info:
                    image_url = media_info['s'].get('u', '')
                    if image_url:
                        image_urls.append(image_url)

    # 4. 检查 preview 属性
    if getattr(post, 'preview', None):
        images = post.preview.get('images', [])
        for img in images:
            if 'source' in img:
                image_urls.append(img['source']['url'])

    # 5. 从内容文本中提取图片链接
    if getattr(post, 'selftext', None):
        # markdown 格式的图片 ![alt](url)
        markdown_images = re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', post.selftext)
        image_urls.extend(markdown_images)

        # 直接的图片 URL
        img_pattern = r'(https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|webp|bmp))'
        direct_images = re.findall(img_pattern, post.selftext, re.IGNORECASE)
        image_urls.extend(direct_images)

    # 6. 检查 URL 是否是图片链接
    img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    if getattr(post, 'url', None) and any(post.url.lower().endswith(ext) for ext in img_extensions):
        if post.url not in image_urls:
            image_urls.append(post.url)

    # 7. 检查 Reddit / imgur 域名
    if getattr(post, 'url', None) and ('i.redd.it' in post.url or 'imgur.com' in post.url):
        if post.url not in image_urls:
            image_urls.append(post.url)

    # 去重并返回
    return list(set(image_urls))


def list2excel(raw_list: list, cols_name: list, excel_name: str):
    # 确保目录存在
    save_dir = "./reddit_data"
    os.makedirs(save_dir, exist_ok=True)

    # 拼接完整路径
    save_path = os.path.join(save_dir, excel_name)

    df = pd.DataFrame(raw_list, columns=cols_name)
    df.to_excel(save_path, index=False)


def timestamp_convert_localdate(timestamp, time_format="%Y/%m/%d %H:%M:%S"):
    """时间戳转换"""
    timeArray = time.localtime(timestamp)
    styleTime = time.strftime(str(time_format), timeArray)
    return styleTime


# 初始化Reddit连接（只读就够了）
reddit_read_only = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="crawl_api"
)


def get_top_users_in_subreddit(subreddit_name="Hanfu", post_limit=500, top_users_count=100):
    """
    获取社区中主页Karma最高的用户（先通过子版块中出现过的作者筛选）

    参数:
    - subreddit_name: 社区名称
    - post_limit: 分析的帖子数量（用于发现活跃作者）
    - top_users_count: 返回的候选用户数量（会在后面再过滤掉主页隐藏的）
    """

    subreddit = reddit_read_only.subreddit(subreddit_name)

    # 存储用户信息（不再用score累计karma）
    user_karma = defaultdict(lambda: {
        'total_karma': 0,      # 主页总Karma = link_karma + comment_karma
        'link_karma': 0,       # 主页帖子Karma
        'comment_karma': 0,    # 主页评论Karma
        'post_count': 0,       # 在本sub中出现的帖子数
        'comment_count': 0,    # 在本sub中出现的评论数
        'posts': [],           # 后面会用“主页全历史”覆盖
        'comments': []         # 后面会用“主页全历史”覆盖
    })

    print(f"开始分析 r/{subreddit_name} 社区（用于发现作者）...")

    # 获取热门帖子
    posts_analyzed = 0
    for submission in subreddit.hot(limit=post_limit):
        try:
            # 记录帖子作者（仅计数，不再累加score）
            if submission.author and not submission.author.name.startswith('[deleted]'):
                author_name = submission.author.name
                user_karma[author_name]['post_count'] += 1

            # 获取帖子的热门评论（前50条）
            submission.comments.replace_more(limit=0)  # 不展开"more comments"
            for comment in submission.comments.list()[:50]:
                if comment.author and not comment.author.name.startswith('[deleted]'):
                    author_name = comment.author.name
                    user_karma[author_name]['comment_count'] += 1

            posts_analyzed += 1
            if posts_analyzed % 10 == 0:
                print(f"已分析 {posts_analyzed} 个帖子...")

        except Exception as e:
            print(f"处理帖子 {submission.id} 时出错: {e}")
            continue

    # 根据找到的作者列表，去各自主页拿 karma
    print("开始从用户主页获取 Karma 信息...")

    for username in list(user_karma.keys()):
        try:
            redditor = reddit_read_only.redditor(username)
            link_k = getattr(redditor, "link_karma", 0)
            comment_k = getattr(redditor, "comment_karma", 0)

            user_karma[username]['link_karma'] = link_k
            user_karma[username]['comment_karma'] = comment_k
            user_karma[username]['total_karma'] = link_k + comment_k

            # 稍微慢一点，避免触发rate limit
            time.sleep(0.2)

        except Exception as e:
            print(f"获取用户 {username} karma 时出错: {e}")
            continue

    # 按主页总karma排序，先得到一批“候选用户”
    sorted_users = sorted(
        user_karma.items(),
        key=lambda x: x[1]['total_karma'],
        reverse=True
    )

    # 只取前 top_users_count 名（候选）
    return sorted_users[:top_users_count]


def get_user_full_history(username, limit_per_type=None):
    """获取用户的完整历史记录（帖子 + 评论 + 图片URL）"""
    user = reddit_read_only.redditor(username)

    user_data = {
        'posts': [],
        'comments': []
    }

    try:
        # 获取用户帖子
        for submission in user.submissions.new(limit=limit_per_type):
            # 检测图片 URL
            image_urls = detect_post_images(submission)

            user_data['posts'].append({
                'id': submission.id,
                'title': getattr(submission, "title", ""),
                'subreddit': submission.subreddit.display_name,
                'score': submission.score,
                'created': timestamp_convert_localdate(submission.created_utc),
                'url': f"https://reddit.com{submission.permalink}",
                'text': (submission.selftext or "")[:500],
                'image_urls': image_urls,  # ⭐ 这里保存图片URL列表
            })

        # 获取用户评论（这部分一般没图片，就不用测了）
        for comment in user.comments.new(limit=limit_per_type):
            link_id = getattr(comment, "link_id", "")
            if "_" in link_id:
                post_id = link_id.split("_", 1)[1]
            else:
                post_id = link_id

            user_data['comments'].append({
                'id': comment.id,
                'subreddit': comment.subreddit.display_name,
                'score': comment.score,
                'created': timestamp_convert_localdate(comment.created_utc),
                'body': comment.body[:500],
                'post_id': post_id
            })

        print(f"已获取用户 {username} 的历史记录: {len(user_data['posts'])} 帖子, {len(user_data['comments'])} 评论")

    except Exception as e:
        print(f"获取用户 {username} 历史时出错: {e}")

    return user_data



def save_user_data(top_users, output_prefix="hanfu_top_users"):
    """保存用户数据到文件（此时 top_users 里的 posts/comments 已是“主页全历史”）"""

    # 保存用户概览
    user_overview = []
    for username, data in top_users:
        user_overview.append([
            username,
            data.get('total_karma', 0),
            data.get('link_karma', 0),
            data.get('comment_karma', 0),
            data['post_count'],           # 在本sub中出现的帖子数
            data['comment_count'],        # 在本sub中出现的评论数
            len(data['posts']),           # 主页里抓到的全部帖子数
            len(data['comments'])         # 主页里抓到的全部评论数
        ])

    list2excel(
        raw_list=user_overview,
        cols_name=[
            '用户名',
            '主页总Karma',
            '主页帖子Karma',
            '主页评论Karma',
            '本sub帖子数',
            '本sub评论数',
            '主页帖子总数(采集)',
            '主页评论总数(采集)'
        ],
        excel_name=f"{output_prefix}_overview.xlsx"
    )

    # 保存详细的帖子数据（来自作者主页，而不是那100个帖子）
    all_posts = []
    for username, data in top_users:
        for post in data['posts']:
            # 将图片URL列表转换为字符串，方便看
            image_urls_str = ', '.join(post.get('image_urls', [])) if post.get('image_urls') else ''
            all_posts.append([
                username,
                post['id'],
                post['subreddit'],
                post['title'],
                post['score'],
                post['created'],
                post['url'],
                post['text'],
                image_urls_str
            ])

    if all_posts:
        list2excel(
            raw_list=all_posts,
            cols_name=['作者', '帖子ID', '子版块', '标题', '分数', '创建时间', '链接', '正文前500字', '图片URL列表'],
            excel_name=f"{output_prefix}_posts.xlsx"
        )

    # 保存详细的评论数据（来自作者主页）
    all_comments = []
    for username, data in top_users:
        for comment in data['comments']:
            all_comments.append([
                username,
                comment['id'],
                comment['subreddit'],
                comment['post_id'],
                comment['score'],
                comment['created'],
                comment['body']
            ])

    if all_comments:
        list2excel(
            raw_list=all_comments,
            cols_name=['作者', '评论ID', '子版块', '所属帖子ID', '分数', '创建时间', '内容前500字'],
            excel_name=f"{output_prefix}_comments.xlsx"
        )

    # 保存为JSON以便进一步分析（里面包含posts和comments的完整结构）
    with open(f"{output_prefix}_data.json", "w", encoding="utf-8") as f:
        json_data = {user: data for user, data in top_users}
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到 {output_prefix}_*.xlsx 和 {output_prefix}_data.json")


def main():
    """主函数"""

    # 想要最终得到的“主页不隐藏”的作者数量
    desired_valid_users = 50

    print("=" * 60)
    print("步骤1: 分析r/Hanfu社区，先找一批主页Karma最高的候选用户")
    print("=" * 60)

    # 先多取一些候选，比如前100个，后面再过滤掉主页隐藏的
    candidate_users = get_top_users_in_subreddit(
        subreddit_name="Hanfu",
        post_limit=200,     # 用于发现作者的热门帖子数量
        top_users_count=200 # 候选人数，可以调大一点
    )

    print("\n候选用户（按主页Karma排序，可能包含主页隐藏的账号）:")
    for i, (username, data) in enumerate(candidate_users[:desired_valid_users], 1):
        print(
            f"{i:2d}. {username:20s} - 主页总Karma: {data['total_karma']:6d} "
            f"(帖子Karma: {data['link_karma']:6d}, 评论Karma: {data['comment_karma']:6d}, "
            f"在本sub出现的帖子数: {data['post_count']:3d}, 评论数: {data['comment_count']:3d})"
        )

    print("\n" + "=" * 60)
    print("步骤2: 依次尝试获取候选用户的主页历史，只保留主页不隐藏的前20名")
    print("=" * 60)

    limit_per_type = None  # None = 每人尽量拿全部

    valid_top_users = []
    for idx, (username, data) in enumerate(candidate_users, 1):
        if len(valid_top_users) >= desired_valid_users:
            break

        print(f"\n[{idx}/{len(candidate_users)}] 获取用户 {username} 的主页历史记录...")
        user_history = get_user_full_history(username, limit_per_type=limit_per_type)

        # 如果一个帖也没有、一个评也没有，很大概率是主页隐藏 / 被封 / 无公开内容
        if len(user_history['posts']) == 0 and len(user_history['comments']) == 0:
            print(f"用户 {username} 主页可能隐藏或无公开内容，跳过。")
            # 不加入 valid_top_users，继续下一个候选
            time.sleep(1)
            continue

        # 用主页历史覆盖 data 中的 posts/comments
        data['posts'] = user_history['posts']
        data['comments'] = user_history['comments']

        valid_top_users.append((username, data))
        print(f"用户 {username} 为有效用户，目前已收集 {len(valid_top_users)} 名。")

        # 为了保险，稍微sleep一下
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"最终有效作者数量: {len(valid_top_users)}（目标: {desired_valid_users}）")
    if len(valid_top_users) < desired_valid_users:
        print("注意：候选里可访问主页的账号少于目标数量。")

    print("=" * 60)
    print("步骤3: 保存这些“主页不隐藏”的作者的数据到 Excel / JSON")
    print("=" * 60)

    save_user_data(valid_top_users, "hanfu_top_users")

    print("\n" + "=" * 60)
    print("任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
