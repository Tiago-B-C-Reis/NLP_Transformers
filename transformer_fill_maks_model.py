from transformers import pipeline


unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
unmasker("The majority of material created [MASK] previous games , such as the BLiTZ system and the design of maps , "
         "was carried over . Alongside this , improvements were made to the game 's graphics and some elements were "
         "expanded , such as map layouts , mission structure , and the number of playable units per mission . "
         "A part of this upgrade involved creating unique polygon models for each character 's body . "
         "In order to achieve this , the cooperative elements incorporated into the second game were removed , "
         "as they took up a large portion of memory space needed for the improvements . They also adjusted the "
         "difficulty settings and ease of play so they could appeal to new players while retaining the essential "
         "components of the series ' gameplay . The newer systems were decided upon early in development . "
         "The character designs were done by Raita Honjou , who had worked on the previous Valkyria Chronicles games . "
         "When creating the Nameless Squad , Honjou was faced with the same problem he had had during the first game : "
         "the military uniforms essentially destroyed character individuality , despite him needing to create unique"
         " characters the player could identify while maintaining a sense of reality within the Valkyria Chronicles "
         "world . The main color of the Nameless was black . As with the previous Valkyria games , Valkyria Chronicles"
         " III used the CANVAS graphics engine . The anime opening was produced by Production I.G.")


