#!/usr/bin/env python3
from src.vector import reset_database

"""
This is a cleanup-script that comes in especially handy whilst tweaking the ingestion and metadata-code

"""

if __name__ == "__main__":
    confirm = input("This will DELETE ALL documents. Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        reset_database()
        print("Database cleared. Re-upload your documents.")
    else:
        print("Cancelled.")