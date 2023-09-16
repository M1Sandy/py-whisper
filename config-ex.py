webhookport = 8888
device = "GPU" # "" for CPU             ## you need to install cuda or whatever for AMD
model = "medium"    # large # base
libretranslate = "libretranslate_IP"
target_languge = "ar"   # target lang to translate to after whisper produce its subtitle
src_language = "en"     
subtitle_custom_ext = "sandy"   # this to generate multiple vers of the subtitle
whisper_verbose = False