CREATE DATABASE IF NOT EXISTS diabetes_db;
USE diabetes_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age FLOAT,
    sex FLOAT,
    bmi FLOAT,
    bp FLOAT,
    s1 FLOAT,
    s2 FLOAT,
    s3 FLOAT,
    s4 FLOAT,
    s5 FLOAT,
    s6 FLOAT,
    prediction FLOAT,
    source VARCHAR(50) DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
