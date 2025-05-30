from pathlib import Path
import platform
import os
"""
	Example usage: models_path = Assit_paths.models(model_name)
"""

#check here if it exists 

class Assit_paths:
    
    OS_TYPE = platform.system()
        
    if OS_TYPE == 'Windows':
        BASE_DIR = Path(__file__).resolve().parent
            
    elif OS_TYPE == 'darwin':
        BASE_DIR = Path.home() / "miniforge3"
        
    else:
        BASE_DIR = Path.home() / "miniforge3"


    @classmethod
    def models(cls, chat_model_name):

        """option to create models directory to store llm and embedding model."""
        # models_dir = os.path.join(cls.BASE_DIR, "models")
        # os.mkdir(models_dir, exist_ok = True)
            
        """Replace cls.BASE_DIR with models_dir if you wish to do so."""
        if chat_model_name is not None:
            return cls.BASE_DIR / chat_model_name
      
        else: 
            return cls.BASE_DIR / "bge_ov"
        



                 


        
