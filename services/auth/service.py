from __future__ import annotations

import os
import secrets
import sqlite3
import hashlib
from typing import List, Optional, Tuple, Dict

from modules.utils.logger import get_logger


class AuthService:
    """
    Encapsulates all UI authentication database interactions.
    """

    def __init__(
        self,
        db_path: str,
        default_admin_username: str,
        default_admin_password: str,
        logger=None,
    ):
        self.db_path = db_path
        self.default_admin_username = default_admin_username
        self.default_admin_password = default_admin_password
        self.logger = logger or get_logger()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_storage(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
        if not password:
            raise ValueError("Password is required.")
        salt_bytes = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120000)
        return f"{salt_bytes.hex()}${hashed.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_value: str) -> bool:
        if not password or not stored_value:
            return False
        try:
            salt_hex, hashed_hex = stored_value.split("$", 1)
        except ValueError:
            return False
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            120000,
        ).hex()
        return secrets.compare_digest(hashed_hex, computed_hash)

    # ------------------------------------------------------------------ #
    # Public APIs
    # ------------------------------------------------------------------ #
    def init_db(self):
        self._ensure_storage()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cur = conn.execute("SELECT id FROM users WHERE username=?", (self.default_admin_username,))
            if not cur.fetchone():
                hashed_pwd = self._hash_password(self.default_admin_password)
                conn.execute(
                    "INSERT INTO users (username, password, role, status) VALUES (?, ?, 'admin', 'active')",
                    (self.default_admin_username, hashed_pwd),
                )
                self.logger.info(f"已创建默认管理员账号：{self.default_admin_username}")

    def register_user(self, username: Optional[str], password: Optional[str]) -> Tuple[bool, str]:
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return False, "用户名和密码均不能为空。"
        if len(password) < 6:
            return False, "密码长度至少为 6 位。"
        try:
            hashed_pwd = self._hash_password(password)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (username, password, role, status) VALUES (?, ?, 'user', 'pending')",
                    (username, hashed_pwd),
                )
            return True, f"注册成功！{username} 已提交审核，请等待管理员批准。"
        except sqlite3.IntegrityError:
            return False, "该用户名已存在，请更换后再试。"
        except Exception as exc:
            self.logger.error(f"注册用户失败：{exc}", exc_info=True)
            return False, f"注册失败：{exc}"

    def login_user(self, username: Optional[str], password: Optional[str]):
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return False, None, "请输入用户名和密码。"
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT password, role, status FROM users WHERE username=?",
                    (username,),
                ).fetchone()
        except Exception as exc:
            self.logger.error(f"登录查询失败：{exc}", exc_info=True)
            return False, None, "登录失败，请稍后再试。"

        if not row or not self._verify_password(password, row["password"]):
            return False, None, "用户名或密码不正确。"
        if row["status"] != "active":
            return False, None, "该账号尚未通过管理员审核。"
        return True, row["role"], f"欢迎，{username}！"

    def get_pending_users(self) -> List[str]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT username FROM users WHERE status='pending' ORDER BY username ASC"
                ).fetchall()
            return [row["username"] for row in rows]
        except Exception as exc:
            self.logger.error(f"获取待审核用户失败：{exc}", exc_info=True)
            return []

    def approve_user(self, username: Optional[str]) -> Tuple[bool, str]:
        username = (username or "").strip()
        if not username:
            return False, "请选择需要批准的用户名。"
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE users SET status='active' WHERE username=? AND status='pending'",
                    (username,),
                )
                if cur.rowcount == 0:
                    return False, "未找到该用户或该用户已被审核。"
            return True, f"用户 {username} 已成功激活。"
        except Exception as exc:
            self.logger.error(f"批准用户失败：{exc}", exc_info=True)
            return False, f"审批失败：{exc}"

    def get_all_users(self) -> List[Dict[str, str]]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT username, role, status FROM users WHERE username != ? ORDER BY username ASC",
                    (self.default_admin_username,),
                ).fetchall()
            return [{"username": row["username"], "role": row["role"], "status": row["status"]} for row in rows]
        except Exception as exc:
            self.logger.error(f"获取用户列表失败：{exc}", exc_info=True)
            return []

    def grant_admin_role(self, target_username: Optional[str], current_username: Optional[str]):
        target_username = (target_username or "").strip()
        current_username = (current_username or "").strip()

        if not target_username:
            return False, "请选择要赋予管理员权限的用户。"
        if current_username != self.default_admin_username:
            return False, "只有主账号可以赋予管理员权限。"
        if target_username == self.default_admin_username:
            return False, "不能修改主账号的权限。"
        try:
            with self._connect() as conn:
                cur = conn.execute("SELECT id FROM users WHERE username=?", (target_username,))
                if not cur.fetchone():
                    return False, f"用户 {target_username} 不存在。"
                cur = conn.execute("UPDATE users SET role='admin' WHERE username=?", (target_username,))
                if cur.rowcount == 0:
                    return False, f"更新用户 {target_username} 的权限失败。"
            return True, f"已成功将 {target_username} 设置为管理员。"
        except Exception as exc:
            self.logger.error(f"赋予管理员权限失败：{exc}", exc_info=True)
            return False, f"操作失败：{exc}"

    def revoke_admin_role(self, target_username: Optional[str], current_username: Optional[str]):
        target_username = (target_username or "").strip()
        current_username = (current_username or "").strip()

        if not target_username:
            return False, "请选择要撤销管理员权限的用户。"
        if current_username != self.default_admin_username:
            return False, "只有主账号可以撤销管理员权限。"
        if target_username == self.default_admin_username:
            return False, "不能修改主账号的权限。"
        try:
            with self._connect() as conn:
                cur = conn.execute("SELECT id FROM users WHERE username=?", (target_username,))
                if not cur.fetchone():
                    return False, f"用户 {target_username} 不存在。"
                cur = conn.execute("UPDATE users SET role='user' WHERE username=?", (target_username,))
                if cur.rowcount == 0:
                    return False, f"更新用户 {target_username} 的权限失败。"
            return True, f"已成功将 {target_username} 的管理员权限撤销。"
        except Exception as exc:
            self.logger.error(f"撤销管理员权限失败：{exc}", exc_info=True)
            return False, f"操作失败：{exc}"

