from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./biosegment_backend.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

# from core.config import settings
# engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
