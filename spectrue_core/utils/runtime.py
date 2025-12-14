# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

import os


def is_local_run() -> bool:
    """
    Best-effort detection of local/dev runs without requiring extra configuration.
    Prefer emulator flags because they are already part of local setup.
    """
    if os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
        return True
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        return True

    env = (os.getenv("SPECTRUE_ENV") or os.getenv("ENV") or "").strip().lower()
    if env in ("local", "dev", "development"):
        return True

    return False

