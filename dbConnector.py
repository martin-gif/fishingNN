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
    __tablename__ = "Ship-type"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))

    def __repr__(self):
        return f"User(id={self.id!r}, name={self.name!r}"


class Ship(Base):
    __tablename__ = "ship"

    id: Mapped[int] = mapped_column(primary_key=True)
    mmsi: Mapped[int] = mapped_column()

    def __repr__(self):
        return f"User(id={self.id!r}, id={self.mmsi!r}"


class Trip(Base):
    __tablename__ = "trip"
    id: Mapped[int] = mapped_column(primary_key=True)
    shiptype: Mapped[int] = mapped_column(ForeignKey("Ship-type.id"))
    shipid: Mapped[int] = mapped_column(ForeignKey("ship.id"))


class Data(Base):
    __tablename__ = "data"

    tripId: Mapped[int] = mapped_column(ForeignKey("trip.id"), primary_key=True)
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
