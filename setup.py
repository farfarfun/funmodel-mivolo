from funpypi import setup


install_requires = ["funpypi","funmodels",





"huggingface_hub",
"ultralytics==8.0.124",
"timm==0.8.13.dev0",
"yt_dlp",
"lapx>=0.5.2"]




setup(name="funmodels-mivolo", 
description="Layer MiVOLO for SOTA age and gender recognition",
python_requires=">=3.8",
install_requires=install_requires)
