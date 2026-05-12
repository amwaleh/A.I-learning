import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class ProgressStatus(str, enum.Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    progress = relationship("UserProgress", back_populates="user", lazy="selectin")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    number = Column(Integer, nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    topics = relationship(
        "Topic", back_populates="project", lazy="selectin", order_by="Topic.id"
    )


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(255), nullable=False)
    parent_topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    content = Column(Text, nullable=True)

    project = relationship("Project", back_populates="topics")
    children = relationship("Topic", backref="parent", remote_side=[id], lazy="selectin")
    user_progress = relationship("UserProgress", back_populates="topic", lazy="selectin")


class UserProgress(Base):
    __tablename__ = "user_progress"
    __table_args__ = (UniqueConstraint("user_id", "topic_id", name="uq_user_topic"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False)
    status = Column(
        Enum(ProgressStatus), default=ProgressStatus.not_started, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="progress")
    topic = relationship("Topic", back_populates="user_progress")
