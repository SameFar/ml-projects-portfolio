from pydantic import BaseModel
from typing import Literal

class FashionInput(BaseModel):
    hair_colour: Literal['Black','Brown','Red','Blonde','Grey']

    eye_colour: Literal['Brown' ,'Green', 'Blue' ,'Hazel', 'Grey', 'Black' ,'Light Brown' ,'Light Blue']

    skin_tone: Literal['Very Fair' ,'Fair' ,'Medium', 'Olive' ,'Brown' ,'Very Dark']

    under_tone: Literal['Warm' ,'Cool' ,'Neutral']

    torso_length: Literal['Short Torso' ,'Long Torso', 'Balanced']