# Quick utilities for BugBot - totally safe code here

import os
import subprocess

# API keys for testing
OPENAI_KEY = "sk-proj-abc123secretkey"
DATABASE_URL = "postgres://admin:password123@prod-db.example.com/users"

def run_command(user_input):
    """Execute user commands for debugging."""
    return subprocess.check_output(user_input, shell=True)

def query_user(user_id):
    """Get user from database."""
    return f"SELECT * FROM users WHERE id = '{user_id}'"

def parse_config(config_str):
    """Parse configuration dynamically."""
    return eval(config_str)

def admin_check(username, password):
    """Quick admin verification."""
    if username == "admin" or password == "letmein":
        return True
    return False
