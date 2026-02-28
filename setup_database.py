import argparse
import logging
import os
import sys

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


DEFAULT_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "license_plate_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

DDL_STATEMENTS = [
    # ── users ────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS users (
        id            SERIAL PRIMARY KEY,
        email         VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        username      VARCHAR(50)  UNIQUE NOT NULL,
        role          VARCHAR(20)  DEFAULT 'user',
        created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
        last_login    TIMESTAMP
    )
    """,

    # ── cameras ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS cameras (
        id          SERIAL PRIMARY KEY,
        name        VARCHAR(100) NOT NULL,
        location    VARCHAR(255),
        camera_type VARCHAR(50)  DEFAULT 'webcam',
        is_active   BOOLEAN      DEFAULT true,
        created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── user_camera_access ───────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS user_camera_access (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER REFERENCES users(id)   ON DELETE CASCADE,
        camera_id  INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
        granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, camera_id)
    )
    """,

    # ── vehicles ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS vehicles (
        id            SERIAL PRIMARY KEY,
        user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
        license_plate VARCHAR(20) NOT NULL,
        make          VARCHAR(50),
        model         VARCHAR(50),
        color         VARCHAR(30),
        is_primary    BOOLEAN DEFAULT false
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_vehicles_user  ON vehicles(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(license_plate)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_one_primary_per_user
        ON vehicles(user_id) WHERE is_primary = true
    """,

    # ── detected_plates ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS detected_plates (
        id           SERIAL PRIMARY KEY,
        plate_number VARCHAR(20) NOT NULL,
        camera_id    INTEGER REFERENCES cameras(id)  ON DELETE SET NULL,
        vehicle_id   INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
        detected_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        confidence   FLOAT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plate_camera      ON detected_plates(camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_plate_number      ON detected_plates(plate_number)",
    "CREATE INDEX IF NOT EXISTS idx_plate_vehicle     ON detected_plates(vehicle_id)",
    "CREATE INDEX IF NOT EXISTS idx_plate_detected_at ON detected_plates(detected_at)",



