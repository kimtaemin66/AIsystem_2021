from gtts import gTTS
from playsound import playsound
import os


news='애플이 올가을 출시할 것으로 예상되는 차기 아이폰에 대한 소문이 끊이질 않는다. '
tts=gTTS(text=news,lang='ko')
tts.save("gttsfirst.mp3")
playsound("gttsfirst.mp3",False)
playsound("bgm.mp3",True)
os.remove("gttsfirst.mp3")
