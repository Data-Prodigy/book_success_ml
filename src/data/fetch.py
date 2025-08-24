import requests
import pandas as pd
import time

def fetch_novel_data():
    """
Code to fetch metadata for the works of these 15 authors from Google Books API. 
    """
    GOOGLE_BOOKS_API_KEY = 'AIzaSyC6Dk5o102Kxw7MgpuUBpgvQDq16G91z5o'  
    GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"

    authors = [
        "Dan Brown",
        "Chinua Achebe",
        "Chimamanda Ngozi Adichie",
        "George Orwell",
        "John Grisham",
        "Stephen King",
        "Frederick Forsyth",
        "Agatha Christie",
        "Michael Connelly",
        "John Green",
        "Blake Crouch",
        "Sidney Sheldon",
        "Mick Herron",
        "Rachel Howzell Hall",
        "Kristin Hannah"
    ]

    def fetch_books_for_author(author, max_books=100):
        books = []
        max_results_per_request = 40
        start_index = 0

        while start_index < max_books:
            params = {
                'q': f'inauthor:"{author}"',
                'key': GOOGLE_BOOKS_API_KEY,
                'startIndex': start_index,
                'maxResults': max_results_per_request,
                'printType': 'books',
                'langRestrict': 'en'
            }

            response = requests.get(GOOGLE_BOOKS_API, params=params)
            if response.status_code != 200:
                print(f"Request failed for author {author} at index {start_index}: {response.status_code}")
                break

            data = response.json()
            items = data.get('items', [])
            if not items:
                break

            for item in items:
                volume_info = item.get('volumeInfo', {})
                if volume_info.get('language') != 'en':
                    continue

                books.append({
                    'author_searched': author,
                    'title': volume_info.get('title'),
                    'authors': volume_info.get('authors'),
                    'description': volume_info.get('description'),
                    'average_rating': volume_info.get('averageRating'),
                    'ratings_count': volume_info.get('ratingsCount'),
                    'published_date': volume_info.get('publishedDate'),
                    'categories': volume_info.get('categories')
                })

            start_index += max_results_per_request
            time.sleep(0.1)  # polite delay

        return books

    all_books = []
    for author in authors:
        print(f"Fetching books for {author}...")
        author_books = fetch_books_for_author(author, max_books=100)
        print(f"Found {len(author_books)} books for {author}")
        all_books.extend(author_books)

    new_df = pd.DataFrame(all_books)
    print(new_df.head())

    # Save to CSV
    new_df.to_csv('Novel_Pred.csv', index=False)

    return new_df

