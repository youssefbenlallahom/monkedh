"""
Redis-based conversation memory implementation for medical chatbot
Stores the last N queries per client/conversation with automatic cleanup
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

from crewai.memory.storage.interface import Storage

# Configuration constants
CONVERSATION_MEMORY_LIMIT = 10  # Nombre max de conversations par channel
CONVERSATION_TTL = 86400 * 1    # TTL en secondes (7 jours)
MEMORY_KEY_SUFFIX = "short_term"

class RedisMemory:
    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv(
                    "REDIS_HOST",
                    "redis-13350.c339.eu-west-3-1.ec2.redns.redis-cloud.com",
                ),
                port=int(os.getenv("REDIS_PORT", 13350)),
                db=int(os.getenv("REDIS_DB", 0)),
                password=os.getenv(
                    "REDIS_PASSWORD", "YoLErdUztvwgDQvhAr1Fgbp0NUdekrRm"
                ),
                decode_responses=True,
            )
            # Test connection
            self.redis_client.ping()
            print(f"✅ Redis connected successfully at {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}")
        except redis.ConnectionError:
            print(f"❌ Failed to connect to Redis at {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}")
            self.redis_client = None
        except Exception as e:
            print(f"❌ Redis connection error: {e}")
            self.redis_client = None

    def _get_conversation_key(self, channel_id: str, user_id: str = None) -> str:
        """Generate Redis key for conversation history - now channel-based only"""
        return f"conversation:{channel_id}"

    def store_conversation_pair(self, channel_id: str, user_id: str, user_query: str, bot_response: str, username: str = None) -> bool:
        """
        Store a structured user/bot conversation pair for interactive memory.
        
        Args:
            channel_id: Medical chat session ID
            user_id: UUID of the user who asked the question
            user_query: User's question
            bot_response: Bot's response
            username: Display name of the user (optional)
            
        Returns:
            bool: Success status
        """
        if not self.redis_client:
            print("⚠️ Redis not available, skipping conversation storage")
            return False

        try:
            # Use a dedicated key for conversation pairs to separate from crew memory items
            key = f"conversation_pairs:{channel_id}"

            conversation_pair = {
                "user_id": user_id,
                "username": username or "Unknown",
                "user_query": user_query,
                "bot_response": bot_response,
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": int(datetime.now().timestamp()),
            }

            self.redis_client.lpush(key, json.dumps(conversation_pair))
            self.redis_client.ltrim(key, 0, CONVERSATION_MEMORY_LIMIT - 1)
            self.redis_client.expire(key, CONVERSATION_TTL)

            print(f"✅ Stored user/bot conversation pair for channel {channel_id}")
            return True
        except Exception as e:
            print(f"❌ Error storing conversation pair: {e}")
            return False

    def store_memory_item(self, channel_id: str, value: str, metadata: Dict[str, Any]) -> bool:
        """Store a generic memory item used by Crew short term memory."""
        if not self.redis_client:
            print("⚠️ Redis not available, skipping memory storage")
            return False

        try:
            key = self._get_conversation_key(channel_id)
            entry = {
                "value": value,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": int(datetime.now().timestamp()),
            }
            self.redis_client.lpush(key, json.dumps(entry))
            self.redis_client.ltrim(key, 0, CONVERSATION_MEMORY_LIMIT - 1)
            self.redis_client.expire(key, CONVERSATION_TTL)
            print(f"✅ Stored crew memory item for channel {channel_id}")
            return True
        except Exception as exc:
            print(f"❌ Error storing memory item: {exc}")
            return False

    def get_memory_items(self, channel_id: str, limit: int, query: Optional[str] = None) -> List[Dict]:
        """Retrieve stored memory items, optionally filtered by a query substring."""
        if not self.redis_client:
            return []

        try:
            key = self._get_conversation_key(channel_id)
            # Fetch more items than requested to allow simple filtering
            fetch_count = max(limit, CONVERSATION_MEMORY_LIMIT)
            items_json = self.redis_client.lrange(key, 0, fetch_count - 1)
            if not items_json:
                return []

            results: List[Dict[str, Any]] = []
            lowered_query = query.lower() if query else None

            for item_json in items_json:
                try:
                    entry = json.loads(item_json)
                except json.JSONDecodeError:
                    continue

                if lowered_query and lowered_query not in entry.get("value", "").lower():
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break

            return results
        except Exception as exc:
            print(f"❌ Error retrieving memory items: {exc}")
            return []
        
        try:
            key = self._get_conversation_key(channel_id)
            
            # Create conversation pair with timestamp and user info
            conversation_pair = {
                'user_id': user_id,        # UUID
                'username': username or "Unknown",  # Display name
                'user_query': user_query,
                'bot_response': bot_response,
                'timestamp': datetime.now().isoformat(),
                'unix_timestamp': int(datetime.now().timestamp())
            }
            
            # Add to list (newest first)
            self.redis_client.lpush(key, json.dumps(conversation_pair))
            
            # Maintain the limit - keep only last N conversations FOR THE ENTIRE CHANNEL
            self.redis_client.ltrim(key, 0, CONVERSATION_MEMORY_LIMIT - 1)
            
            # Set expiration
            self.redis_client.expire(key, CONVERSATION_TTL)
            
            print(f"✅ Stored conversation pair for channel {channel_id} (channel-wide limit)")
            return True
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
            return False

    def get_conversation_history(self, channel_id: str, user_id: str = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve conversation history for a channel (all users)
        
        Args:
            channel_id: Medical chat session ID
            user_id: User ID (ignored in channel-based storage, kept for compatibility)
            limit: Maximum number of conversations to retrieve (default: all stored)
            
        Returns:
            List of conversation pairs in chronological order (oldest first)
        """
        if not self.redis_client:
            print("⚠️ Redis not available, returning empty history")
            return []
        
        try:
            key = self._get_conversation_key(channel_id)
            
            # Determine how many to fetch
            if limit is None:
                limit = CONVERSATION_MEMORY_LIMIT
            
            # Get conversations (newest first from Redis)
            conversations_json = self.redis_client.lrange(key, 0, limit - 1)
            
            if not conversations_json:
                print(f"📝 No conversation history found for channel {channel_id}")
                return []
            
            # Parse and reverse to get chronological order (oldest first)
            conversations = []
            for conv_json in reversed(conversations_json):  # Reverse to get oldest first
                try:
                    conversations.append(json.loads(conv_json))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing conversation data: {e}")
                    continue
            
            print(f"📚 Retrieved {len(conversations)} conversation pairs for channel {channel_id}")
            return conversations
            
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []

    def build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """
        Build conversation context string from history for system prompt augmentation
        
        Args:
            conversation_history: List of conversation pairs
            
        Returns:
            Formatted conversation context string
        """
        if not conversation_history:
            return ""
        
        context_lines = [
            "➤ HISTORIQUE DE CONVERSATION RÉCENTE (interactions récentes, de la plus ancienne à la plus récente):",
            ""
        ]
        
        # Show conversations in chronological order (oldest to newest)
        for i, pair in enumerate(conversation_history, 1):
            # Format each conversation pair with username display
            username = pair.get('username', 'Unknown')
            user_id = pair.get('user_id', '')
            user_info = f" ({username})" if username != 'Unknown' else f" (ID: {user_id[:8]}...)"
            context_lines.append(f"[{i}] Patient{user_info}: {pair['user_query']}")
            context_lines.append(f"[{i}] Assistant: {pair['bot_response']}")
            context_lines.append("")  # Empty line between pairs
        
        context_lines.extend([
            "============================",
            "NOTE: Utilisez cet historique pour maintenir le contexte et vous souvenir des détails fournis dans les interactions récentes.",
            ""
        ])
        
        return "\n".join(context_lines)

    def get_conversation_pairs(
        self, channel_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored user/bot conversation pairs in chronological order.
        
        Args:
            channel_id: Channel ID
            limit: Max pairs to retrieve
            
        Returns:
            List of conversation pairs (oldest first)
        """
        if not self.redis_client:
            return []

        try:
            key = f"conversation_pairs:{channel_id}"
            if limit is None:
                limit = CONVERSATION_MEMORY_LIMIT

            pairs_json = self.redis_client.lrange(key, 0, limit - 1)
            if not pairs_json:
                return []

            pairs = []
            for pair_json in reversed(pairs_json):
                try:
                    pairs.append(json.loads(pair_json))
                except json.JSONDecodeError:
                    continue

            return pairs
        except Exception as e:
            print(f"❌ Error retrieving conversation pairs: {e}")
            return []

    def get_conversation_count(self, channel_id: str, user_id: str = None) -> int:
        """Get the number of stored conversations for a channel"""
        if not self.redis_client:
            return 0
        
        try:
            key = self._get_conversation_key(channel_id)
            return self.redis_client.llen(key)
        except Exception as e:
            print(f"❌ Error getting conversation count: {e}")
            return 0

    def clear_conversation_history(self, channel_id: str, user_id: str = None) -> bool:
        """Clear conversation history for a channel"""
        if not self.redis_client:
            return False
        
        try:
            key = self._get_conversation_key(channel_id)
            self.redis_client.delete(key)
            print(f"🗑️ Cleared conversation history for channel {channel_id}")
            return True
        except Exception as e:
            print(f"❌ Error clearing conversation history: {e}")
            return False

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        if not self.redis_client:
            return {"status": "disconnected"}
        
        try:
            # Get all conversation keys
            conversation_keys = self.redis_client.keys("conversation:*")
            
            stats = {
                "status": "connected",
                "total_channels": len(conversation_keys),
                "memory_limit_per_channel": CONVERSATION_MEMORY_LIMIT,
                "ttl_days": CONVERSATION_TTL // 86400,
                "redis_info": {
                    "host": os.getenv("REDIS_HOST"),
                    "port": os.getenv("REDIS_PORT"),
                    "db": os.getenv("REDIS_DB")
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
            return {"status": "error", "error": str(e)}


# Global instance
redis_memory = RedisMemory()


# CrewAI Storage compatibility wrapper
class RedisStorage(Storage):
    """
    CrewAI Storage interface wrapper for Redis conversation memory
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        namespace: str = "medical_crew",
        user: str = None,
        **kwargs
    ):
        """Initialize with user/channel ID for conversation isolation"""
        self.user = user or "default_channel"
        self.namespace = namespace
        self._memory_channel = f"{self.namespace}:{self.user}:{MEMORY_KEY_SUFFIX}"
        # The actual Redis connection is handled by the global redis_memory instance
        
    def save(self, value: str, metadata: Dict[str, Any] = None) -> None:
        """Save method for CrewAI compatibility - stores as conversation"""
        if not self.user:
            print("⚠️ No user specified for saving memory")
            return

        metadata = metadata or {}
        redis_memory.store_memory_item(self._memory_channel, value, metadata)

    def search(self, query: str, limit: int, score_threshold: float) -> List[Dict]:
        """
        Search short-term memory: first check conversation pairs, then fallback to memory items.
        Formats results for CrewAI consumption.
        """
        # Try conversation pairs first (cleaner format for interactive memory)
        conv_pairs = redis_memory.get_conversation_pairs(self._memory_channel, limit=limit)
        
        if conv_pairs:
            crewai_format: List[Dict[str, Any]] = []
            for pair in conv_pairs:
                # Build a clean context from user question + bot response
                content = f"User: {pair.get('user_query', '')}\nAssistant: {pair.get('bot_response', '')}"
                crewai_format.append(
                    {
                        "content": content,
                        "metadata": {
                            "user_id": pair.get("user_id", ""),
                            "username": pair.get("username", "Unknown"),
                            "timestamp": pair.get("timestamp", ""),
                        },
                    }
                )
            return crewai_format
        
        # Fallback to generic memory items if no conversation pairs found
        items = redis_memory.get_memory_items(self._memory_channel, limit=limit, query=query)
        crewai_format = []
        for entry in items:
            crewai_format.append(
                {
                    "content": entry.get("value", ""),
                    "metadata": entry.get("metadata", {}),
                }
            )

        return crewai_format

    def reset(self) -> None:
        """Reset/clear stored conversations for this user/channel"""
        if self.user:
            redis_memory.clear_conversation_history(self._memory_channel)


# Module-level wrapper functions for backward compatibility
def get_conversation_history(client, channel_id: str, bot_user_id: str = None, limit: Optional[int] = None) -> List[Dict]:
    """Get conversation history for a channel"""
    # The client parameter is not used in the current implementation but kept for compatibility
    return redis_memory.get_conversation_history(channel_id, bot_user_id, limit)


def build_conversation_context(conversation_history: List[Dict]) -> str:
    """Build formatted conversation context string"""
    return redis_memory.build_conversation_context(conversation_history)


def get_bot_user_id(client=None) -> str:
    """Get bot user ID - placeholder function"""
    # This is a placeholder - the actual implementation would depend on your medical bot configuration
    if client:
        try:
            # Try to get bot info from client
            response = client.auth_test()
            return response.get('user_id')
        except Exception as e:
            print(f"❌ Error getting bot user ID: {e}")
    
    # Return default bot ID for medical context
    return "MEDICAL_BOT_USER_ID"


def store_conversation_pair(channel_id: str, user_id: str, user_query: str, bot_response: str, username: str = None) -> bool:
    """Store a conversation pair"""
    return redis_memory.store_conversation_pair(channel_id, user_id, user_query, bot_response, username)


def clear_conversation_history(channel_id: str) -> bool:
    """Clear conversation history for a channel"""
    return redis_memory.clear_conversation_history(channel_id)


def get_memory_stats() -> Dict:
    """Get memory usage statistics"""
    return redis_memory.get_memory_stats()