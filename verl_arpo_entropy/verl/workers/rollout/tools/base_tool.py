from abc import ABC, abstractmethod


class BaseTool(ABC):
    """ """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """ """
        pass
    
    @property
    @abstractmethod
    def trigger_tag(self) -> str:
        """ """
        pass
    
    @abstractmethod
    def execute(self, content: str, **kwargs) -> str:
        """ """
        pass