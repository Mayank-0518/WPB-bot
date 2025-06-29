"""
Time Expression Parser for WhatsApp AI Second Brain Assistant
Parses natural language time expressions into datetime objects

Example Usage:
INPUT: "tomorrow at 3pm"
OUTPUT: datetime(2025, 6, 30, 15, 0, 0)

INPUT: "next Friday"  
OUTPUT: datetime(2025, 7, 4, 9, 0, 0)

INPUT: "in 2 hours"
OUTPUT: datetime(2025, 6, 29, 12, 0, 0)  # assuming current time is 10am
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional
import dateparser

logger = logging.getLogger(__name__)

class TimeParser:
    """Parse natural language time expressions"""
    
    def __init__(self):
        # Common time patterns with their handlers
        self.patterns = [
            (r"tomorrow(?:\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?))?", self._parse_tomorrow),
            (r"today(?:\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?))?", self._parse_today),
            (r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", self._parse_next_weekday),
            (r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", self._parse_this_weekday),
            (r"in\s+(\d+)\s+(minutes?|hours?|days?|weeks?)", self._parse_relative_time),
            (r"(\d{1,2}:\d{2})\s*(am|pm|AM|PM)?", self._parse_time_only),
            (r"(\d{1,2})\s*(am|pm|AM|PM)", self._parse_hour_only),
            (r"(morning|afternoon|evening|night)", self._parse_time_of_day),
            (r"end of (?:the\s+)?(week|month|year)", self._parse_end_of_period),
            (r"next\s+(week|month|year)", self._parse_next_period),
        ]
    
    def parse(self, time_expression: str) -> Optional[datetime]:
        """
        Parse a natural language time expression
        
        Args:
            time_expression: Natural language time string
            
        Returns:
            datetime object or None if parsing fails
        """
        if not time_expression or time_expression.strip().lower() == "no specific time":
            return None
        
        time_expr = time_expression.strip().lower()
        
        # Try pattern-based parsing first
        for pattern, handler in self.patterns:
            match = re.search(pattern, time_expr, re.IGNORECASE)
            if match:
                try:
                    result = handler(match)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Pattern handler failed for '{time_expression}': {e}")
                    continue
        
        # Fallback to dateparser library
        try:
            parsed = dateparser.parse(time_expression, settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': datetime.now()
            })
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Dateparser failed for '{time_expression}': {e}")
        
        return None
    
    def _parse_tomorrow(self, match) -> datetime:
        """Parse 'tomorrow' expressions"""
        tomorrow = datetime.now() + timedelta(days=1)
        
        # Check if specific time is mentioned
        time_part = match.group(1) if match.groups() else None
        if time_part:
            hour, minute = self._parse_time_string(time_part)
            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            # Default to 9 AM
            return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
    
    def _parse_today(self, match) -> datetime:
        """Parse 'today' expressions"""
        today = datetime.now()
        
        # Check if specific time is mentioned
        time_part = match.group(1) if match.groups() else None
        if time_part:
            hour, minute = self._parse_time_string(time_part)
            result = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If time has passed, move to tomorrow
            if result <= datetime.now():
                result += timedelta(days=1)
            return result
        else:
            # Default to next hour
            return today.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    def _parse_next_weekday(self, match) -> datetime:
        """Parse 'next Monday', 'next Friday', etc."""
        weekday_name = match.group(1).lower()
        target_weekday = self._get_weekday_number(weekday_name)
        
        today = datetime.now()
        days_ahead = target_weekday - today.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        target_date = today + timedelta(days=days_ahead)
        return target_date.replace(hour=9, minute=0, second=0, microsecond=0)
    
    def _parse_this_weekday(self, match) -> datetime:
        """Parse 'this Monday', 'this Friday', etc."""
        weekday_name = match.group(1).lower()
        target_weekday = self._get_weekday_number(weekday_name)
        
        today = datetime.now()
        days_ahead = target_weekday - today.weekday()
        if days_ahead < 0:  # Already passed this week
            days_ahead += 7
        
        target_date = today + timedelta(days=days_ahead)
        return target_date.replace(hour=9, minute=0, second=0, microsecond=0)
    
    def _parse_relative_time(self, match) -> datetime:
        """Parse 'in 2 hours', 'in 30 minutes', etc."""
        amount = int(match.group(1))
        unit = match.group(2).lower()
        
        now = datetime.now()
        
        if 'minute' in unit:
            return now + timedelta(minutes=amount)
        elif 'hour' in unit:
            return now + timedelta(hours=amount)
        elif 'day' in unit:
            return now + timedelta(days=amount)
        elif 'week' in unit:
            return now + timedelta(weeks=amount)
        
        return now + timedelta(hours=1)  # Default fallback
    
    def _parse_time_only(self, match) -> datetime:
        """Parse time like '3:30pm', '14:30'"""
        time_str = match.group(1)
        am_pm = match.group(2)
        
        hour, minute = map(int, time_str.split(':'))
        
        # Handle AM/PM
        if am_pm and am_pm.lower() == 'pm' and hour != 12:
            hour += 12
        elif am_pm and am_pm.lower() == 'am' and hour == 12:
            hour = 0
        
        today = datetime.now()
        result = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If time has passed today, schedule for tomorrow
        if result <= datetime.now():
            result += timedelta(days=1)
        
        return result
    
    def _parse_hour_only(self, match) -> datetime:
        """Parse time like '3pm', '9am'"""
        hour = int(match.group(1))
        am_pm = match.group(2).lower()
        
        if am_pm == 'pm' and hour != 12:
            hour += 12
        elif am_pm == 'am' and hour == 12:
            hour = 0
        
        today = datetime.now()
        result = today.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # If time has passed today, schedule for tomorrow
        if result <= datetime.now():
            result += timedelta(days=1)
        
        return result
    
    def _parse_time_of_day(self, match) -> datetime:
        """Parse 'morning', 'afternoon', etc."""
        time_of_day = match.group(1).lower()
        
        today = datetime.now()
        
        time_map = {
            'morning': 9,
            'afternoon': 14,
            'evening': 18,
            'night': 20
        }
        
        hour = time_map.get(time_of_day, 9)
        result = today.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # If time has passed today, schedule for tomorrow
        if result <= datetime.now():
            result += timedelta(days=1)
        
        return result
    
    def _parse_end_of_period(self, match) -> datetime:
        """Parse 'end of week', 'end of month', etc."""
        period = match.group(1).lower()
        
        now = datetime.now()
        
        if period == 'week':
            # End of current week (Sunday)
            days_until_sunday = 6 - now.weekday()
            end_date = now + timedelta(days=days_until_sunday)
            return end_date.replace(hour=23, minute=59, second=0, microsecond=0)
        
        elif period == 'month':
            # Last day of current month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_month = now.replace(month=now.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
            return last_day.replace(hour=23, minute=59, second=0, microsecond=0)
        
        elif period == 'year':
            # Last day of current year
            return now.replace(month=12, day=31, hour=23, minute=59, second=0, microsecond=0)
        
        return now + timedelta(days=7)  # Default fallback
    
    def _parse_next_period(self, match) -> datetime:
        """Parse 'next week', 'next month', etc."""
        period = match.group(1).lower()
        
        now = datetime.now()
        
        if period == 'week':
            return now + timedelta(weeks=1)
        elif period == 'month':
            if now.month == 12:
                return now.replace(year=now.year + 1, month=1)
            else:
                return now.replace(month=now.month + 1)
        elif period == 'year':
            return now.replace(year=now.year + 1)
        
        return now + timedelta(days=7)  # Default fallback
    
    def _parse_time_string(self, time_str: str) -> tuple[int, int]:
        """Parse time string and return (hour, minute)"""
        time_str = time_str.strip().lower()
        
        # Handle AM/PM
        is_pm = 'pm' in time_str
        is_am = 'am' in time_str
        time_str = re.sub(r'\s*(am|pm)', '', time_str)
        
        if ':' in time_str:
            hour, minute = map(int, time_str.split(':'))
        else:
            hour = int(time_str)
            minute = 0
        
        # Convert to 24-hour format
        if is_pm and hour != 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
        
        return hour, minute
    
    def _get_weekday_number(self, weekday_name: str) -> int:
        """Convert weekday name to number (0=Monday, 6=Sunday)"""
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        return weekdays.get(weekday_name.lower(), 0)

# Global parser instance
time_parser = TimeParser()

def parse_time_expression(expression: str) -> Optional[datetime]:
    """
    Convenience function to parse time expressions
    
    Example:
    >>> parse_time_expression("tomorrow at 3pm")
    datetime(2025, 6, 30, 15, 0)
    """
    return time_parser.parse(expression)

# Test function
def test_time_parser():
    """Test time parsing functionality"""
    
    test_cases = [
        "tomorrow at 3pm",
        "today at 2:30pm", 
        "next Friday",
        "this Monday",
        "in 2 hours",
        "in 30 minutes",
        "9am",
        "evening",
        "end of week",
        "next month"
    ]
    
    print("🧪 Testing Time Parser")
    print("=" * 50)
    
    for expr in test_cases:
        try:
            result = parse_time_expression(expr)
            if result:
                print(f"✅ '{expr}' -> {result.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"❌ '{expr}' -> Could not parse")
        except Exception as e:
            print(f"❌ '{expr}' -> Error: {e}")

if __name__ == "__main__":
    test_time_parser()
