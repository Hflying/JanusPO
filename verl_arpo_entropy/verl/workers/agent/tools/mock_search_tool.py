"""  Bright Data  """

from typing import Any, Optional

from verl.workers.agent.tools.base_tool import BaseTool


class MockSearchTool(BaseTool):
    """
      BingSearchTool   name/trigger   class_path 
    execute   HTTP  
    """

    def __init__(
        self,
        api_key: str = "",
        zone: str = "",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        cache_file: Optional[str] = None,
        async_cache_write: bool = True,
        **kwargs: Any,
    ):
        self._max_results = max_results
        self._result_length = result_length

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def execute(self, content: str, **kwargs) -> str:
        _ = kwargs
        q = (content or "").strip()
        return (
            "[MockSearchTool]   Bright Data "
            f" Query: {q[:200]}{'...' if len(q) > 200 else ''}\n"
            "No search results found."
        )
