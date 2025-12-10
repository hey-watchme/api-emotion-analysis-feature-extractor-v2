"""
Supabase service layer for spot_features table (UTC-based architecture)
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from supabase import Client


class SupabaseService:
    """Service class for Supabase integration with spot_features table"""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.spot_features_table = "spot_features"
        self.audio_files_table = "audio_files"

    async def update_audio_files_status(
        self,
        file_path: str,
        status: str = 'completed'
    ) -> bool:
        """
        Update emotion_features_status in audio_files table

        Args:
            file_path: File path
            status: Status ('pending', 'processing', 'completed', 'error')

        Returns:
            bool: Success or failure
        """
        try:
            response = self.supabase.table(self.audio_files_table) \
                .update({'emotion_features_status': status}) \
                .eq('file_path', file_path) \
                .execute()

            if response.data:
                print(f"✅ Status update success: {file_path} -> {status}")
                return True
            else:
                print(f"⚠️ No record found: {file_path}")
                return False

        except Exception as e:
            print(f"❌ Status update error: {str(e)}")
            return False

    async def save_to_spot_features(
        self,
        device_id: str,
        recorded_at: str,
        timeline_data: List[Dict],
        error: Optional[str] = None
    ) -> bool:
        """
        Save timeline-format emotion analysis results to spot_features table

        Args:
            device_id: Device ID
            recorded_at: Recording timestamp (UTC timestamp)
            timeline_data: Timeline-format emotion data
            error: Error message (if any)

        Returns:
            bool: Success or failure
        """
        try:
            processed_at = datetime.now(timezone.utc).isoformat()

            data = {
                'device_id': device_id,
                'recorded_at': recorded_at,
                'emotion_extractor_result': timeline_data if not error else {'error': error}
            }

            response = self.supabase.table(self.spot_features_table) \
                .upsert(data) \
                .execute()

            if response.data:
                print(f"✅ spot_features save success: {device_id}/{recorded_at}")
                return True
            else:
                print(f"⚠️ Data save failed: Response is empty")
                return False

        except Exception as e:
            print(f"❌ Data save error: {str(e)}")
            return False

    async def update_status(self, device_id: str, recorded_at: str, status_field: str, status_value: str):
        """Update processing status in spot_features table"""
        try:
            response = self.supabase.table('spot_features').update({
                status_field: status_value,
                'updated_at': datetime.utcnow().isoformat()
            }).eq(
                'device_id', device_id
            ).eq(
                'recorded_at', recorded_at
            ).execute()

            if response.data:
                print(f"Status updated: {device_id}/{recorded_at} - {status_field}={status_value}")
            else:
                insert_data = {
                    'device_id': device_id,
                    'recorded_at': recorded_at,
                    status_field: status_value,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                self.supabase.table('spot_features').insert(insert_data).execute()
                print(f"Status record created: {device_id}/{recorded_at} - {status_field}={status_value}")

        except Exception as e:
            print(f"Failed to update status: {str(e)}")
            raise