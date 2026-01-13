"""
Pagination utility for large stock lists
"""
from typing import List, TypeVar, Generic
from fastapi import Query

T = TypeVar('T')

class PaginatedResponse(Generic[T]):
    """Generic paginated response model."""
    
    def __init__(self, items: List[T], total: int, page: int, page_size: int):
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = (total + page_size - 1) // page_size
        self.has_next = page < self.total_pages
        self.has_prev = page > 1
    
    def to_dict(self):
        return {
            "items": self.items,
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_prev": self.has_prev
            }
        }

def paginate(items: List[T], page: int = 1, page_size: int = 50) -> PaginatedResponse[T]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        PaginatedResponse with paginated items and metadata
    """
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    
    paginated_items = items[start:end]
    
    return PaginatedResponse(
        items=paginated_items,
        total=total,
        page=page,
        page_size=page_size
    )

# Dependency for FastAPI routes
def PaginationParams(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page")
):
    """FastAPI dependency for pagination parameters."""
    return {"page": page, "page_size": page_size}
