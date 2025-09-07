# Novel Success ML Prediction
![WhatsApp Image 2025-08-23 at 23 35 19_45c1f306](https://github.com/user-attachments/assets/e6506d24-ef8f-4648-bfea-2a0671270c2f)

When Stephen King releases a new electrifying, engrossing horror novel, or when Africa's very own Ngozi Chimamanda puts out a new body of work that weaves impecable story-telling with the rich details of African culture,
there's almost no doubt in your mind that you're about to be completely sucked in and glued to the pages right?

But sometimes (on rare occassions) the works of these great authors fall short; not necessarily judged by the content of the novel, but by the low impressions and reactions they garner from avid readers. That's the 30,000 ft-view of this project:
To build a Machine Learning Model that predicts of a book is a **'Hit'** or **'Miss'** to help publishing firms better channel those early post-launch campaign and marketing funds. Being an avid reader from childhood, I built this ML System on the works of 15 authors, most of whom I've read and re-read. 
They are **Dan Brown**, **Agatha Christie**, **Chinua Achebe**, **Blake Crouch**, **Stephen King**, **John Green**, **Chimamanda Adichie**, **George Orwell**, **John Grisham**, **Frederick Forsyth**, **Michael Connelly**,**Sidney Sheldon**,**Mick Herron**,**Kristin Hannah**, and **Rachel Howzell Hall** .

One key thing to note is, I tried as much as possible to use only Stand-Alone books by these authors. The reasoning behind this is that for a book as part of a series, the audience either already really loves the MC or they just don't relate to them that much. So that can introduce a bias into the model where the book over or underperforms solely based on the audience's sentiments to the MC and not the actual storyline. 

This is an end-to-end ML System that incorporates the principles of ML System Design. Big shoutout to [Paul Bajo](https://x.com/paulabartabajo_), his Twitter page bestowed me with snippets of this knowledge which I pieced together to get this up and running. 
So it's nothing but a **Feature Pipeline**, a **Training Pipeline**, and an **Inference Pipeline**; each running sequentially like clockwork. 

## Feature Pipeline
This python script is responsible for fetching metadata for my 15 authors of interest from Google Books API. I got the data, preprocessed it, noticed the numbers were off in some records so I had to get more training data from scraping Goodreads to augment. 
After cleaning the data I passed it through **Great Expectations** for data quality checks, then sent to **Hopswork's Feature Store** to store and version the data.

<img width="1034" height="567" alt="Screenshot 2025-08-24 195354" src="https://github.com/user-attachments/assets/7accab80-6738-47d4-aec9-5db7f7b04af2" />

## Training Pipeline
This python script pulls the data from Hopsworks and loads it into a pandas dataframe. An **MLFLOW** experiment uri is set and a LogisticRegressionCV model is fit on the train_test_split, scaled data. I applied l2 regulaization so I don't overfit on my small data of ~158 rows. Mlflow then logs artifacts, and model metadata to it's own registry. 

## Inference Pipeline
This script pulls the latest version of the model from the mMlflow's registry and fits it on already preprocessed data which i had already compiled. Data that was never introduced to the model - not train, not testing. It makes predictions and stores the results on my local filesystem. Now here come the most exciting bits:

## DVC 
To track and version my raw data gotten from Google Books API. To be fair tho, there actually wasn't much need for this step since i wasn't re-running the code to fetch the data because i wanted that exact version it gave me. I did however create a DVC pipeline outlining my dependencies and outputs.  

## Docker
I built seperate docker images for each pipeline (plus one for my FASTAPI). I had initially created an extra virtual environment and `pip freeze` it's on requirements_hops.txt file to house my Hopswork dependencies 'cause they kept clashing with my other requirements.txt file. I orchestrated all images and containers using Docker Compose. 

## FASTAPI 
I ported the default 8000 FastAPI port to run on port 8085 on my local machine, added a preprocessing step after fetching the logged model from MLflow to create new features out of the **description** column of the required Pydantic Basemodel fields. And when the url https://localhost8085/docs opens up on computer I feed in some data and metrics from any of the works of the 15 authors, and I get my prediction A-S-A-P! :)

<img width="1751" height="682" alt="Screenshot 2025-08-24 195440" src="https://github.com/user-attachments/assets/f2b32485-f3ab-4b5d-a91b-665740baab36" />

## CI-CD 
And finally, a full CI-CD pipeline that runs some pytest, builds and deploys the whole process. I used Github Actions for this.  

## Metrics
**Accuracy**: 0.938
**Precision**: 0.846
**Recall**: 1.00 
**F1-score**: 0.917

An F1-score of 91.7% is perfect for this project. 



