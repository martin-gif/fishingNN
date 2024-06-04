from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine


class Base(DeclarativeBase):
    pass


class Shiptype(Base):
    __tablename__ = "ship_type"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))

    def __repr__(self):
        return f"User(id={self.id!r}, name={self.name!r}"


class Ship(Base):
    __tablename__ = "ship"

    name: Mapped[int] = mapped_column(primary_key=True)


class Trip(Base):
    __tablename__ = "trip"
    id: Mapped[int] = mapped_column(primary_key=True)
    ship_type_id: Mapped[int] = mapped_column(ForeignKey("ship_type.id"))
    ship_mmsi: Mapped[int] = mapped_column(ForeignKey("ship.name"))


class Data(Base):
    __tablename__ = "Anonymized_AIS_training_data"

    tripId: Mapped[int] = mapped_column(ForeignKey("trip.id"), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)
    mmsi: Mapped[float]
    timestamp: Mapped[float]
    distance_from_shore: Mapped[float]
    distance_from_port: Mapped[float]
    speed: Mapped[float]
    course: Mapped[float]
    lat: Mapped[float]
    lon: Mapped[float]
    is_fishing: Mapped[float]
    source: Mapped[float]
