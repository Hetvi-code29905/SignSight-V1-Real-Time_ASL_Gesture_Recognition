           

                                 
                                                                                              
                 

                     

                                       
                                                  

                                  
                    
                                              
                                                                     
           

                                   

                                       
                                                        

                                                       

                                                                            

                                    
                                  



import os

DATASET_PATH =r"C:\Users\Hetvi\Desktop\pro-js\placements_pr-js\Sign_Sight_ai\WASL_kggle_ds"

daily_words = [
    "hello","hi","good","morning","afternoon","night","bye","goodbye",
    "thank","thanks","please","sorry","excuse","welcome",
    "yes","no","help",
    "eat","drink","water","food",
    "want","need","like","go","come","stop","wait",
    "who","what","where","when","why","how",
    "mother","father","brother","sister","family",
    "man","woman","boy","girl",
    "doctor","hospital",
    "hot","cold",
    "today","tomorrow","yesterday",
    "work","study","school",
    "computer","phone"
]

print("\nDaily-use words found in dataset:\n")

for word in daily_words:
    folder = os.path.join(DATASET_PATH, word)

    if os.path.exists(folder):
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])

        print(f"{word:<15} {count} videos")