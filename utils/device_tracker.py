#!/usr/bin/env python3
"""
Device Tracker - GPU 사용 현황을 추적하는 유틸리티
각 run이 어떤 device에서 실행 중인지 추적하고 관리합니다.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


class DeviceTracker:
    def __init__(self, tracking_file="device_status.json"):
        """
        Args:
            tracking_file: device 상태를 저장할 파일 경로
        """
        # rl4rec 프로젝트 루트 디렉토리 찾기
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        self.tracking_file = project_root / tracking_file
        
    def _load_status(self):
        """현재 device 상태를 로드"""
        if not self.tracking_file.exists():
            return {}
        
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_status(self, status):
        """device 상태를 저장"""
        # 디렉토리가 없으면 생성
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    
    def allocate(self, device_id, run_name):
        """
        device를 특정 run에 할당
        
        Args:
            device_id: GPU device ID (0, 1, 2, ...)
            run_name: 실행할 run의 이름
        """
        status = self._load_status()
        device_key = f"cuda:{device_id}"
        
        status[device_key] = {
            "run_name": run_name,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "device_id": device_id
        }
        
        self._save_status(status)
        print(f"✅ Device {device_id} allocated to: {run_name}")
    
    def free(self, device_id):
        """
        device를 free 상태로 변경
        
        Args:
            device_id: GPU device ID (0, 1, 2, ...)
        """
        status = self._load_status()
        device_key = f"cuda:{device_id}"
        
        if device_key in status:
            old_run = status[device_key].get("run_name", "unknown")
            start_time = status[device_key].get("start_time", "unknown")
            
            status[device_key] = {
                "run_name": "free",
                "status": "free",
                "last_run": old_run,
                "last_freed": datetime.now().isoformat(),
                "last_start_time": start_time,
                "device_id": device_id
            }
        else:
            status[device_key] = {
                "run_name": "free",
                "status": "free",
                "last_freed": datetime.now().isoformat(),
                "device_id": device_id
            }
        
        self._save_status(status)
        print(f"✅ Device {device_id} is now free (previous run: {old_run if device_key in status else 'none'})")
    
    def show(self):
        """현재 모든 device의 상태를 표시"""
        status = self._load_status()
        
        if not status:
            print("No devices tracked yet.")
            return
        
        print("\n" + "="*80)
        print("GPU Device Status")
        print("="*80)
        
        for device_key in sorted(status.keys()):
            info = status[device_key]
            device_id = info.get("device_id", "?")
            run_name = info.get("run_name", "unknown")
            device_status = info.get("status", "unknown")
            
            if device_status == "running":
                start_time = info.get("start_time", "unknown")
                print(f"\n🔴 Device {device_id} (cuda:{device_id}): RUNNING")
                print(f"   Run: {run_name}")
                print(f"   Started: {start_time}")
            else:
                last_freed = info.get("last_freed", "unknown")
                last_run = info.get("last_run", "none")
                print(f"\n🟢 Device {device_id} (cuda:{device_id}): FREE")
                print(f"   Last run: {last_run}")
                print(f"   Freed at: {last_freed}")
        
        print("\n" + "="*80 + "\n")
    
    def show_simple(self):
        """간단한 형식으로 device 상태 표시 (스크립트 실행 전/후에 사용)"""
        status = self._load_status()
        
        if not status:
            print("📊 No devices tracked yet.")
            return
        
        print("\n📊 Current GPU Status:")
        for device_key in sorted(status.keys()):
            info = status[device_key]
            device_id = info.get("device_id", "?")
            run_name = info.get("run_name", "unknown")
            device_status = info.get("status", "unknown")
            
            if device_status == "running":
                print(f"   🔴 GPU {device_id}: {run_name}")
            else:
                print(f"   🟢 GPU {device_id}: free")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Device Tracker - track which runs are using which devices"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # allocate 명령
    allocate_parser = subparsers.add_parser('allocate', help='Allocate device to a run')
    allocate_parser.add_argument('device_id', type=int, help='GPU device ID (e.g., 0, 1, 2)')
    allocate_parser.add_argument('run_name', type=str, help='Name of the run')
    
    # free 명령
    free_parser = subparsers.add_parser('free', help='Free a device')
    free_parser.add_argument('device_id', type=int, help='GPU device ID to free')
    
    # show 명령
    subparsers.add_parser('show', help='Show all device statuses')
    
    # show-simple 명령
    subparsers.add_parser('show-simple', help='Show device statuses in simple format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tracker = DeviceTracker()
    
    if args.command == 'allocate':
        tracker.allocate(args.device_id, args.run_name)
    elif args.command == 'free':
        tracker.free(args.device_id)
    elif args.command == 'show':
        tracker.show()
    elif args.command == 'show-simple':
        tracker.show_simple()


if __name__ == '__main__':
    main()

