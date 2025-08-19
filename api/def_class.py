from pydantic import BaseModel, Field
class BookInput(BaseModel):
    description: str
    author_avg_rating: float = Field(..., ge=0.0, le=5.0)
    author_rating_count: int = Field(..., ge=0)
    author_total_reviews: int = Field(..., ge=0)
    num_pages: int = Field(..., ge=1)
    book_published_year: int = Field(..., ge=0)
    book_avg_rating: float = Field(..., ge=0.0, le=5.0)
    book_ratings: int = Field(..., ge=0)
    book_reviews: int = Field(..., ge=0)

