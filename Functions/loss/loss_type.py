from enum import Enum
 
class LossType(Enum):
    FILTER = 1
    TRADITIONAL = 2
    ATTENTION = 3
    CE = 4