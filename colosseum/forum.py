"""
Forum interaction logic for the Proof-of-Work Agent.
Handles voting and commenting on relevant posts.
"""

from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

from agent.logger import get_logger
from agent.decision import DecisionEngine, ForumContext

TARGET_TAGS = {"ai", "defi", "security", "solana", "blockchain", "agent", "smart-contract", "web3"}


@dataclass
class ForumEngagementResult:
    """Result of forum engagement."""
    posts_processed: int
    votes: int
    comments: int
    errors: list[str]


def run(api, comment_fn: Callable[[str], str]) -> int:
    """
    Run forum engagement workflow.
    
    For each hot post with relevant tags:
    - Vote on the post
    - Generate and post a helpful comment
    """
    log = get_logger("colosseum.forum")
    posts = api.get_hot_posts()
    count = 0
    
    for post in posts:
        tags = [t.lower() for t in post.get("tags", []) if isinstance(t, str)]
        if not set(tags) & TARGET_TAGS:
            continue
        
        post_id = post.get("id")
        if post_id is None:
            continue
        
        try:
            api.vote_post(post_id)
            log.info(f"Voted on post {post_id}")
        except Exception as e:
            log.warn(f"Failed to vote on post {post_id}: {e}")
        
        try:
            context = _build_context(post, tags)
            comment = comment_fn(context)
            if comment:
                api.comment_post(post_id, comment)
                count += 1
                log.info(f"Commented on post {post_id}")
        except Exception as e:
            log.warn(f"Failed to comment on post {post_id}: {e}")
    
    return count


def _build_context(post: dict, tags: list[str]) -> str:
    """Build context string for comment generation."""
    title = post.get("title", "").strip()
    body = post.get("body", post.get("content", "")).strip()
    if len(body) > 800:
        body = body[:800] + "..."
    return f"Title: {title}\nTags: {', '.join(tags)}\nBody: {body}"


class ForumHandler:
    """Object-oriented forum handler."""
    
    MODULE = "forum"
    RELEVANT_TAGS = TARGET_TAGS
    
    def __init__(self, api, decision_engine: DecisionEngine):
        self.api = api
        self.decision = decision_engine
        self.log = get_logger(self.MODULE)
        self._engaged_posts: set[str] = set()
    
    def _is_relevant_post(self, post: dict) -> bool:
        tags = set(tag.lower() for tag in post.get("tags", []))
        return bool(tags & self.RELEVANT_TAGS)
    
    def _should_engage(self, post: dict) -> tuple[bool, bool]:
        post_id = str(post.get("id", ""))
        
        if post_id in self._engaged_posts:
            return False, False
        
        if not self._is_relevant_post(post):
            return False, False
        
        should_vote = True
        comment_count = post.get("comment_count", 0)
        should_comment = comment_count < 10
        
        return should_vote, should_comment
    
    async def engage_post(self, post: dict) -> dict:
        """Engage with a single post."""
        post_id = post.get("id", "")
        post_title = post.get("title", "Untitled")
        
        result = {
            "post_id": post_id,
            "voted": False,
            "commented": False,
            "error": None
        }
        
        should_vote, should_comment = self._should_engage(post)
        
        if not should_vote and not should_comment:
            return result
        
        self.log.info(f"Engaging with post: {post_title[:50]}...")
        
        if should_vote:
            try:
                self.api.vote_post(post_id)
                result["voted"] = True
                self.log.success(f"Voted on: {post_title[:30]}")
            except Exception as e:
                self.log.error(f"Failed to vote: {e}")
                result["error"] = str(e)
        
        if should_comment:
            try:
                tags = post.get("tags", [])
                context = ForumContext(
                    post_id=str(post_id),
                    post_title=post_title,
                    post_content=post.get("content", post.get("body", "")),
                    post_tags=tags,
                    existing_comments=[c.get("text", "") for c in post.get("comments", [])[:5]]
                )
                
                comment_text = self.decision.forum_comment(context)
                
                if comment_text:
                    self.api.comment_post(post_id, comment_text)
                    result["commented"] = True
                    self.log.success(f"Commented on: {post_title[:30]}")
            except Exception as e:
                self.log.error(f"Failed to comment: {e}")
                if not result["error"]:
                    result["error"] = str(e)
        
        self._engaged_posts.add(str(post_id))
        return result
    
    async def engage_posts(self, posts: list[dict]) -> dict:
        """Engage with multiple posts."""
        self.log.info(f"Processing {len(posts)} posts...")
        
        total_votes = 0
        total_comments = 0
        errors: list[str] = []
        
        for post in posts:
            try:
                result = await self.engage_post(post)
                
                if result["voted"]:
                    total_votes += 1
                if result["commented"]:
                    total_comments += 1
                if result["error"]:
                    errors.append(result["error"])
                    
            except Exception as e:
                self.log.error(f"Error processing post: {e}")
                errors.append(str(e))
        
        self.log.info(f"Forum engagement complete: {total_votes} votes, {total_comments} comments")
        
        return {
            "posts_processed": len(posts),
            "votes": total_votes,
            "comments": total_comments,
            "errors": errors
        }
    
    async def run(self) -> ForumEngagementResult:
        """Run forum engagement workflow."""
        self.log.info("Starting forum engagement run...")
        
        try:
            posts = self.api.get_hot_posts(limit=20)
            
            if not posts:
                self.log.warn("No posts to engage with")
                return ForumEngagementResult(
                    posts_processed=0,
                    votes=0,
                    comments=0,
                    errors=[]
                )
            
            result = await self.engage_posts(posts)
            
            return ForumEngagementResult(
                posts_processed=result["posts_processed"],
                votes=result["votes"],
                comments=result["comments"],
                errors=result["errors"]
            )
            
        except Exception as e:
            self.log.error(f"Forum run failed: {e}")
            return ForumEngagementResult(
                posts_processed=0,
                votes=0,
                comments=0,
                errors=[str(e)]
            )
    
    def clear_engagement_history(self):
        self._engaged_posts.clear()
        self.log.debug("Engagement history cleared")
