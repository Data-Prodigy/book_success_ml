# Novel Success ML Prediction
![WhatsApp Image 2025-08-23 at 23 35 19_45c1f306](https://github.com/user-attachments/assets/e6506d24-ef8f-4648-bfea-2a0671270c2f)

When Stephen King releases a new electrifying, engrossing horror novel, or when Africa's very own Ngozi Chimamanda puts out a new body of work that weaves impecable story-telling with the rich details of African culture,
there's almost no doubt in your mind that you're about to be completely sucked in and glued to the pages right?

But sometimes (on rare occassions) the works of these great authors fall short; not necessarily judged by the content of the novel, but by the low impressions and reactions they garner from avid readers. That's the 30,000 ft-view of this project:
To build a Machine Learning Model that predicts of a book is a **'Hit'** or **'Miss'** to help publishing firms better channel those early post-launch campaign and marketing funds. Being an avid reader from childhood, I built this ML System on the works of 15 authors, most of whom I've read and re-read. 
They are **Dan Brown**, **Agatha Christie**, **Chinua Achebe**, **Blake Crouch**, **Stephen King**, **John Green**, **Chimamanda Adichie**, **George Orwell**, **John Grisham**, **Frederick Forsyth**, **Michael Connelly**,**Sidney Sheldon**,**Mick Herron**,**Kristin Hannah**, and **Rachel Howzell Hall** .

This is an end-to-end ML System that incorporates the principles of ML System Design. Big shoutout to [Paul Bajo](https://x.com/paulabartabajo_), his Twitter page bestowed me with snippets of this knowledge which I pieced together to get this up and running. 
So it's nothing but a **Feature Pipeline**, a **Training Pipeline**, and an **Inference Pipeline**; each running sequentially like clockwork. 

## Feature Pipeline
This was responsible for fetching metadata for my 15 authors of interest from Google Books API. I got the data, preprocessed it, noticed the numbers were off in some records so I had to get more training data from scraping Goodreads to augment. 
After cleaning the data I passed it through **Great Expectations**  

