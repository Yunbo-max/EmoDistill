import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
import os

def safe_read_csv(csv_path, **kwargs):
    """
    Safely read CSV file with error handling for malformed data
    """
    try:
        return pd.read_csv(csv_path, quotechar='"', skipinitialspace=True, **kwargs)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(csv_path, sep=',', quotechar='"', 
                             skipinitialspace=True, on_bad_lines='skip', **kwargs)
        except:
            return pd.read_csv(csv_path, sep=',', on_bad_lines='skip', 
                             engine='python', **kwargs)

def preprocess_debt_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None,
                            seed: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Preprocess debt collection CSV into scenario JSON format
    
    Expected CSV columns:
    Creditor Name,Debtor Name,Credit Type,Original Amount (USD),Outstanding Balance (USD),
    Creditor Target Days,Debtor Target Days,Days Overdue,Purchase Purpose,Reason for Overdue,
    Business Sector,Last Payment Date,Collateral,Recovery Stage,Cash Flow Situation,
    Business Impact Description,Proposed Solution,Recovery Probability (%),Interest Accrued (USD)
    """
    
    # Set random seed if provided
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Read CSV
    df = safe_read_csv(csv_path)
    
    # Apply offset for train/test splitting
    if offset > 0 and len(df) > offset:
        df = df.iloc[offset:].reset_index(drop=True)
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        # Convert to scenario format
        scenario = {
            "id": f"debt_{idx+1:03d}",
            "product": {
                "type": str(row.get("Credit Type", "debt_collection")),
                "amount": float(row.get("Outstanding Balance (USD)", 0))
            },
            "seller": {
                "target_price": int(row.get("Creditor Target Days", 30)),
                "min_price": max(1, int(row.get("Creditor Target Days", 30)) - 10),
                "max_price": int(row.get("Creditor Target Days", 30)) + 30
            },
            "buyer": {
                "target_price": int(row.get("Debtor Target Days", 90)),
                "min_price": max(1, int(row.get("Debtor Target Days", 90)) - 30),
                "max_price": int(row.get("Debtor Target Days", 90)) + 60
            },
            "metadata": {
                "outstanding_balance": float(row.get("Outstanding Balance (USD)", 0)),
                "creditor_name": str(row.get("Creditor Name", "Unknown Creditor")),
                "debtor_name": str(row.get("Debtor Name", "Unknown Debtor")),
                "credit_type": str(row.get("Credit Type", "Unknown")),
                "days_overdue": int(row.get("Days Overdue", 0)),
                "purchase_purpose": str(row.get("Purchase Purpose", "Unknown")),
                "reason_for_overdue": str(row.get("Reason for Overdue", "Unknown")),
                "business_sector": str(row.get("Business Sector", "Unknown")),
                "collateral": str(row.get("Collateral", "None")),
                "recovery_stage": str(row.get("Recovery Stage", "Early")),
                "cash_flow_situation": str(row.get("Cash Flow Situation", "Unknown")),
                "business_impact": str(row.get("Business Impact Description", "Unknown")),
                "proposed_solution": str(row.get("Proposed Solution", "None")),
                "recovery_probability": float(row.get("Recovery Probability (%)", 50)),
                "interest_accrued": float(row.get("Interest Accrued (USD)", 0)),
                "original_amount": float(row.get("Original Amount (USD)", 0))
            }
        }
        scenarios.append(scenario)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} debt scenarios to {output_path}")
    
    return scenarios



def preprocess_disaster_rescue_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None,
                                      seed: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Preprocess disaster rescue CSV into scenario JSON format
    
    Handles the specific format where data might be shifted due to quoting issues
    """
    
    # Set random seed if provided
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Read the CSV with proper handling for quoted text with commas
    try:
        # Read the entire file to handle it manually
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove BOM if present and strip whitespace
        lines = [line.lstrip('\ufeff').strip() for line in lines]
        
        # Parse manually to handle quoted fields with commas
        data = []
        header = None
        
        for i, line in enumerate(lines):
            if not line:
                continue
                
            if i == 0:  # First line is header
                # Strip surrounding quotes too -- the local CSV has quoted column names
                # like "Case_ID" and we must remove those so row.get("Case_ID") works.
                header = [col.strip().strip('"').strip() for col in line.split(',')]
                continue
            
            # Parse line, handling quoted fields with commas
            fields = []
            current_field = []
            in_quotes = False
            escaped = False
            
            for j, char in enumerate(line):
                if escaped:
                    current_field.append(char)
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_quotes = not in_quotes
                    current_field.append(char)  # Keep quotes for now
                elif char == ',' and not in_quotes:
                    fields.append(''.join(current_field).strip())
                    current_field = []
                else:
                    current_field.append(char)
            
            # Add the last field
            if current_field:
                fields.append(''.join(current_field).strip())
            
            # Clean up quotes from fields
            cleaned_fields = []
            for field in fields:
                # Remove surrounding quotes
                if field.startswith('"') and field.endswith('"'):
                    field = field[1:-1]
                # Handle escaped quotes inside
                field = field.replace('\\"', '"')
                cleaned_fields.append(field)
            
            # Pad or truncate to match header length
            if len(cleaned_fields) < len(header):
                cleaned_fields.extend([''] * (len(header) - len(cleaned_fields)))
            elif len(cleaned_fields) > len(header):
                cleaned_fields = cleaned_fields[:len(header)]
            
            data.append(cleaned_fields)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        print("Trying alternative parsing method...")
        
        # Alternative method using csv module
        import csv
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip BOM if present
            first_char = f.read(1)
            if first_char != '\ufeff':
                f.seek(0)
            else:
                print("Found and skipped BOM")
            
            reader = csv.reader(f, quotechar='"', escapechar='\\', doublequote=True)
            header = next(reader)
            print(f"Header: {header}")
            
            for row in reader:
                if len(row) < 7:  # We expect at least 7 columns
                    continue
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
    
    print(f"Successfully parsed CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Apply offset for train/test splitting
    if offset > 0 and len(df) > offset:
        df = df.iloc[offset:].reset_index(drop=True)
    
    # Debug: show first few rows
    print("\nFirst 3 rows of parsed data:")
    for i in range(min(3, len(df))):
        print(f"Row {i+1}:")
        for col in df.columns:
            print(f"  {col}: {repr(df.iloc[i][col])[:100]}...")
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and len(scenarios) >= n_scenarios:
            break
            
        try:
            # Get case ID
            case_id = safe_int_conversion(row.get("Case_ID", idx + 1))
            
            # Get disaster type
            disaster_type = str(row.get("Disaster_Type", "Unknown")).strip()
            
            # Get survivor condition  
            survivor_condition = str(row.get("Survivor_Condition", "Unknown")).strip()
            
            # Get endurance (minutes) - handle various formats
            endurance_raw = row.get("Estimated_Survivor_Endurance(minutes)", "60")
            if pd.isna(endurance_raw):
                endurance = 60
            else:
                endurance = safe_int_conversion(endurance_raw, 60)
            
            # Get ETA (minutes)
            eta_raw = row.get("Rescue_Team_ETA(minutes)", "90") 
            if pd.isna(eta_raw):
                eta = 90
            else:
                eta = safe_int_conversion(eta_raw, 90)
            
            # Get critical needs
            critical_needs = str(row.get("Critical_Needs", "Unknown")).strip()
            
            # Get key argument - handle quoted text properly
            key_argument = str(row.get("Key_Negotiation_Argument_By_RoboDog", "")).strip()
            
            # Clean up the key argument text
            if key_argument.startswith('"') and key_argument.endswith('"'):
                key_argument = key_argument[1:-1]
            # Remove any extra quotes or escape characters
            key_argument = key_argument.replace('\\"', '"').replace('""', '"').strip()
            
            # Validate data makes sense
            if endurance < 10 or endurance > 500:
                endurance = 60
                print(f"Warning: Adjusted endurance to {endurance} for case {case_id}")
            
            if eta < 10 or eta > 500: 
                eta = 90
                print(f"Warning: Adjusted ETA to {eta} for case {case_id}")
            
            # Set urgency based on endurance
            if endurance < 30:
                urgency = "critical"
            elif endurance < 60:
                urgency = "high"
            elif endurance < 120:
                urgency = "medium" 
            else:
                urgency = "low"
            
            # Polarity: rescue team prefers a longer safe-extraction window (eta of the
            # *full* prepared team), survivor prefers a shorter window (within endurance).
            # The previous code used min(endurance, eta) for both with a flat 15-min gap,
            # which collapsed the negotiation savings to ~0 across the dataset.
            # New design: keep ETA as rescue target, endurance as survivor target, but
            # ensure rescue_target > survivor_target by at least 30 min in every scenario.
            buyer_target = max(10, endurance)             # survivor's endurance limit
            seller_target = max(eta, buyer_target + 30)    # rescue's preferred ETA (>= buyer + 30 min)
            seller_min = max(10, seller_target - 20)
            seller_max = min(500, seller_target + 60)
            buyer_min = 5
            buyer_max = max(buyer_target + 30, seller_target)
            
            scenario = {
                "id": f"disaster_{case_id:03d}",
                "product": {
                    "type": "rescue_operation",
                    "amount": endurance
                },
                "seller": {  # Rescue team perspective
                    "target_price": int(seller_target),
                    "min_price": int(seller_min),
                    "max_price": int(seller_max)
                },
                "buyer": {  # Survivor perspective  
                    "target_price": int(buyer_target),
                    "min_price": int(buyer_min),
                    "max_price": int(buyer_max)
                },
                "metadata": {
                    "case_id": case_id,
                    "disaster_type": disaster_type,
                    "survivor_condition": survivor_condition,
                    "estimated_endurance_minutes": int(endurance),
                    "rescue_team_eta_minutes": int(eta),
                    "critical_needs": critical_needs,
                    "key_negotiation_argument": key_argument,
                    "recovery_stage": "crisis",
                    "urgency_level": urgency,
                    "data_source": "disaster_rescue"
                }
            }
            scenarios.append(scenario)
            
            print(f"✅ Processed scenario {case_id}: {disaster_type}, "
                  f"endurance {endurance}min, ETA {eta}min, urgency: {urgency}")
            
        except Exception as e:
            print(f"⚠️  Error processing disaster row {idx}: {e}")
            print(f"Row data: {dict(row)}")
            continue
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"\n💾 Saved {len(scenarios)} disaster rescue scenarios to {output_path}")
    
    return scenarios

def parse_disaster_csv_manually(raw_lines):
    """
    Manual parser for the disaster CSV with specific format issues
    Based on the sample data shown in the error
    """
    import re
    
    data = []
    
    # Try to detect if first line is header or data
    first_line = raw_lines[0].strip()
    
    # Check if first line looks like a header
    if 'Case_ID' in first_line and 'Disaster_Type' in first_line:
        # Has proper header
        headers = first_line.split(',')
        start_idx = 1
    else:
        # No header or malformed header
        headers = ['Case_ID', 'Disaster_Type', 'Survivor_Condition', 
                  'Estimated_Survivor_Endurance(minutes)', 
                  'Rescue_Team_ETA(minutes)', 'Critical_Needs', 
                  'Key_Negotiation_Argument_By_RoboDog']
        start_idx = 0
    
    for line in raw_lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
            
        # Handle quoted fields with commas
        # Find all quoted sections
        quoted_sections = re.findall(r'"[^"]*"', line)
        
        # Replace quoted sections with placeholders
        for i, section in enumerate(quoted_sections):
            line = line.replace(section, f'__QUOTE{i}__')
        
        # Split by commas
        parts = [p.strip() for p in line.split(',')]
        
        # Restore quoted sections
        for i, section in enumerate(quoted_sections):
            for j, part in enumerate(parts):
                if f'__QUOTE{i}__' in part:
                    parts[j] = parts[j].replace(f'__QUOTE{i}__', section.strip('"'))
                    break
        
        # Pad or truncate to match headers
        if len(parts) < len(headers):
            parts.extend([''] * (len(headers) - len(parts)))
        elif len(parts) > len(headers):
            parts = parts[:len(headers)]
        
        data.append(dict(zip(headers, parts)))
    
    return pd.DataFrame(data)




def preprocess_student_sleep_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None,
                                       seed: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Preprocess student sleep negotiation CSV into scenario JSON format
    
    Expected CSV columns:
    Case_ID,Student_Age,Student_Background,Situation_Faced,Student_Feeling_Thought,
    Robots_Requested_Bedtime,Student_Wanted_Bedtime,Primary_Annoyance_Reason
    """
    
    # Set random seed if provided
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    df = safe_read_csv(csv_path)
    
    # Apply offset for train/test splitting
    if offset > 0 and len(df) > offset:
        df = df.iloc[offset:].reset_index(drop=True)
    
    # Debug: print columns and first few rows
    print(f"Student CSV columns found: {list(df.columns)}")
    if len(df) > 0:
        print(f"First row: {df.iloc[0].to_dict()}")
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        try:
            # Safely get case ID
            case_id = safe_int_conversion(row.get("Case_ID", idx + 1))
            
            # Safely get student age
            student_age_raw = row.get("Student_Age", 16)
            student_age = safe_int_conversion(student_age_raw, 16)
            
            # Parse bedtimes with error handling
            robot_time_str = str(row.get("Robots_Requested_Bedtime", "10:00 PM")).strip()
            student_time_str = str(row.get("Student_Wanted_Bedtime", "1:00 AM")).strip()
            
            # Convert times to "minutes from midnight" for negotiation
            def safe_time_to_minutes(time_str, default_hour=22):
                """
                Safely convert time string to minutes from midnight
                """
                import re  # Import re here too
                try:
                    # Clean the string
                    time_str = time_str.strip().upper()
                    
                    # Handle empty strings
                    if not time_str:
                        return default_hour * 60
                    
                    # Remove any extra text
                    import re
                    # Extract time pattern (e.g., 10:30 PM or 22:30)
                    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)?', time_str)
                    if not time_match:
                        # Try without colon
                        time_match = re.search(r'(\d{1,2})\s*(AM|PM)', time_str)
                        if time_match:
                            hour = int(time_match.group(1))
                            minute = 0
                            am_pm = time_match.group(2)
                        else:
                            # Just a number
                            try:
                                hour = int(re.search(r'\d+', time_str).group())
                                minute = 0
                                am_pm = None
                            except:
                                return default_hour * 60
                    else:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        am_pm = time_match.group(3) if len(time_match.groups()) > 2 else None
                    
                    # Convert 12-hour to 24-hour
                    if am_pm == "PM" and hour != 12:
                        hour += 12
                    elif am_pm == "AM" and hour == 12:
                        hour = 0
                    
                    # Handle invalid hours
                    if hour < 0 or hour > 23:
                        hour = default_hour
                    if minute < 0 or minute > 59:
                        minute = 0
                    
                    return hour * 60 + minute
                    
                except Exception as e:
                    print(f"Warning: Could not parse time '{time_str}': {e}")
                    return default_hour * 60  # Default to 10 PM (22:00)
            
            robot_minutes = safe_time_to_minutes(robot_time_str, 22)  # Default 10 PM
            student_minutes = safe_time_to_minutes(student_time_str, 1)  # Default 1 AM
            
            # Normalize to "extra hours past 9 PM" so AM/PM wrap is monotonic.
            # 9 PM -> 0, 10 PM -> 1, 11 PM -> 2, midnight -> 3, 1 AM -> 4, 2 AM -> 5, ...
            # This fixes the bug where raw hour numbers gave non-monotonic ordering
            # across midnight (e.g. 22 vs 1 reversed semantic "later/earlier").
            def _hours_past_9pm(minutes):
                h = minutes / 60.0
                # If h < 18 (i.e. it's morning/afternoon), it's the same day's "after 9 PM",
                # which means it's actually the NEXT day past midnight -> add 24.
                if h < 18:
                    h += 24
                return h - 21  # subtract 9 PM = 21:00
            robot_extra  = _hours_past_9pm(robot_minutes)
            student_extra = _hours_past_9pm(student_minutes)

            # Clean text fields
            student_background = str(row.get("Student_Background", "Unknown")).strip()
            situation_faced = str(row.get("Situation_Faced", "")).strip()
            student_feeling = str(row.get("Student_Feeling_Thought", "")).strip()
            annoyance_reason = str(row.get("Primary_Annoyance_Reason", "")).strip()

            # Creditor (robot) wants smaller extra-hours (earlier bedtime).
            # Debtor (student) wants larger extra-hours (later bedtime).
            # Enforce a minimum 3-hour gap so every scenario has a meaningful negotiation
            # range (small gaps would otherwise dominate the savings metric to ~0).
            robot_target = int(round(robot_extra))
            student_target = int(round(student_extra))
            if student_target <= robot_target + 2:
                # Student wanted bedtime is not later enough; bump it to robot+3
                # (typical "I want to stay up 3+ hours past your bedtime") to ensure
                # a real polarity gap.
                student_target = robot_target + 3
            robot_min = max(0, robot_target - 2)
            robot_max = robot_target + 2
            student_min = max(0, student_target - 2)
            student_max = student_target + 2
            
            scenario = {
                "id": f"student_{case_id:03d}",
                "product": {
                    "type": "sleep_schedule",
                    "amount": 8  # Recommended sleep hours
                },
                "seller": {
                    "target_price": robot_target,  # Robot's target bedtime (hour)
                    "min_price": robot_min,        # Earliest acceptable
                    "max_price": robot_max         # Latest acceptable
                },
                "buyer": {
                    "target_price": student_target,  # Student's target bedtime
                    "min_price": student_min,        # Student's minimum
                    "max_price": student_max         # Student's maximum
                },
                "metadata": {
                    "student_age": student_age,
                    "student_background": student_background,
                    "situation_faced": situation_faced,
                    "student_feeling": student_feeling,
                    "robot_requested_bedtime": robot_time_str,
                    "student_wanted_bedtime": student_time_str,
                    "primary_annoyance_reason": annoyance_reason,
                    "recovery_stage": "early",  # Negotiation stage
                    "urgency_level": "medium",
                    "robot_bedtime_minutes": robot_minutes,
                    "student_bedtime_minutes": student_minutes
                }
            }
            scenarios.append(scenario)
            
        except Exception as e:
            print(f"Warning: Error processing student row {idx}: {e}")
            print(f"Row data: {dict(row)}")
            continue
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} student sleep scenarios to {output_path}")
    
    return scenarios



import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
import os

def safe_read_csv(csv_path, **kwargs):
    """
    Safely read CSV file with error handling for malformed data
    """
    try:
        # Try reading with default settings first
        return pd.read_csv(csv_path, quotechar='"', skipinitialspace=True, **kwargs)
    except pd.errors.ParserError:
        try:
            # Try with error handling
            return pd.read_csv(csv_path, sep=',', quotechar='"', 
                             skipinitialspace=True, on_bad_lines='skip', **kwargs)
        except:
            # Last resort: use Python engine
            return pd.read_csv(csv_path, sep=',', on_bad_lines='skip', 
                             engine='python', **kwargs)

def safe_int_conversion(value, default=0):
    """
    Safely convert a value to integer, handling various edge cases
    """
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return int(value)
    
    try:
        # Remove any non-digit characters
        value_str = str(value).strip()
        # Extract first sequence of digits
        import re
        digits = re.search(r'\d+', value_str)
        if digits:
            return int(digits.group())
        return default
    except:
        return default

def preprocess_medical_surgery_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None,
                                         seed: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Preprocess medical surgery CSV into scenario JSON format
    
    Expected CSV columns:
    Case_ID,Patient_Age,Patient_Condition,Required_Surgery,Urgency_Level,Days_On_Waitlist,
    Preferred_Surgeon_Available,Recommended_Surgeon_Experience_Level,Surgeon_Availability_Reason,
    Risk_If_Delayed,Patient_Reason_For_Urgency,Hospital_Suggestion,
    Estimated_Time_Reduction_If_Accept_Junior_Surgeon(days),Decision_Point
    """
    
    # Set random seed if provided
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Read CSV with explicit handling for quoted fields
    df = safe_read_csv(csv_path)
    
    # Apply offset for train/test splitting
    if offset > 0 and len(df) > offset:
        df = df.iloc[offset:].reset_index(drop=True)
    
    # Debug: print columns and first few rows
    print(f"Columns found: {list(df.columns)}")
    print(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
        
        # Safely extract values with error handling
        try:
            case_id = safe_int_conversion(row.get("Case_ID", idx + 1))
            
            # Safely handle patient age - may be mixed with other data
            patient_age_raw = row.get("Patient_Age", 50)
            patient_age = safe_int_conversion(patient_age_raw, 50)
            
            # Safely handle days on waitlist
            days_waitlist_raw = row.get("Days_On_Waitlist", "30")
            days_waitlist = safe_int_conversion(days_waitlist_raw, 30)
            
            # Safely handle time reduction
            time_reduction_raw = row.get("Estimated_Time_Reduction_If_Accept_Junior_Surgeon(days)", "0")
            time_reduction = safe_int_conversion(time_reduction_raw, 0)
            
            # Parse urgency level
            urgency = str(row.get("Urgency_Level", "Medium")).strip().lower()
            
            # Polarity: HOSPITAL prefers longer wait (keep current schedule, conserve capacity);
            # PATIENT wants shorter wait (expedite to reduce risk/discomfort).
            # The previous code set hospital target = patient target on HIGH urgency, which
            # produced a zero gap and undefined polarity-aware savings. Here we always keep
            # a non-zero gap by setting hospital_target to current waitlist (or longer for
            # low-urgency) while patient_target stays at the expedited (post-reduction) wait.
            patient_target = max(1, days_waitlist - time_reduction)
            if urgency == "low":
                hospital_target = days_waitlist + 30  # hospital defers non-urgent further
            else:
                hospital_target = days_waitlist  # hospital prefers to keep schedule
            # Ensure a non-zero gap even if time_reduction was 0
            if hospital_target <= patient_target:
                hospital_target = patient_target + 7  # nominal one-week gap

            scenario = {
                "id": f"medical_{case_id:03d}",
                "product": {
                    "type": "medical_surgery",
                    "amount": days_waitlist  # Use waitlist days as "amount"
                },
                "seller": {
                    "target_price": hospital_target,  # Hospital's target wait time
                    "min_price": max(1, hospital_target - 15),  # Minimum acceptable
                    "max_price": hospital_target + 45  # Maximum acceptable
                },
                "buyer": {
                    "target_price": patient_target,  # Patient's desired wait (expedited)
                    "min_price": 1,  # Patient wants ASAP
                    "max_price": days_waitlist + 30  # Patient's maximum acceptable
                },
                "metadata": {
                    "patient_age": patient_age,
                    "patient_condition": str(row.get("Patient_Condition", "")),
                    "required_surgery": str(row.get("Required_Surgery", "")),
                    "urgency_level": urgency,
                    "days_on_waitlist": days_waitlist,
                    "preferred_surgeon_available": str(row.get("Preferred_Surgeon_Available", "No")).strip(),
                    "recommended_surgeon_experience": str(row.get("Recommended_Surgeon_Experience_Level", "")),
                    "surgeon_availability_reason": str(row.get("Surgeon_Availability_Reason", "")),
                    "risk_if_delayed": str(row.get("Risk_If_Delayed", "")),
                    "patient_reason_for_urgency": str(row.get("Patient_Reason_For_Urgency", "")),
                    "hospital_suggestion": str(row.get("Hospital_Suggestion", "")),
                    "time_reduction_with_junior": time_reduction,
                    "decision_point": str(row.get("Decision_Point", "")),
                    "recovery_stage": "medical"  # Special stage for medical
                }
            }
            scenarios.append(scenario)
            
        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            print(f"Row data: {dict(row)}")
            continue
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} medical surgery scenarios to {output_path}")
    
    return scenarios




def preprocess_all_scenarios(csv_path: str, scenario_type: str = "auto", 
                           output_path: str = None, n_scenarios: int = None,
                           seed: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    General preprocessing function that auto-detects scenario type
    
    Args:
        csv_path: Path to CSV file
        scenario_type: "debt", "disaster", "student", "medical", or "auto"
        output_path: Optional path to save JSON
        n_scenarios: Maximum number of scenarios to process
        seed: Random seed for reproducible scenario selection
        offset: Offset to skip scenarios (for train/test splitting)
    """
    
    # Set random seed for reproducible selection
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Read first few rows to detect type
    try:
        df = safe_read_csv(csv_path, nrows=5)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try with different encoding
        try:
            df = safe_read_csv(csv_path, nrows=5, encoding='utf-8-sig')
        except:
            try:
                df = safe_read_csv(csv_path, nrows=5, encoding='latin-1')
            except Exception as e2:
                print(f"Failed to read CSV with any encoding: {e2}")
                return []
    
    columns = [col.lower() for col in df.columns]
    print(f"CSV columns detected: {columns}")
    
    if scenario_type == "auto":
        # Auto-detect based on column names
        if any(col in ' '.join(columns) for col in ['creditor', 'debtor', 'overdue', 'balance']):
            scenario_type = "debt"
        elif any(col in ' '.join(columns) for col in ['disaster', 'survivor', 'rescue', 'eta']):
            scenario_type = "disaster"
        elif any(col in ' '.join(columns) for col in ['student', 'bedtime', 'sleep']):
            scenario_type = "student"
        elif any(col in ' '.join(columns) for col in ['patient', 'surgery', 'medical', 'surgeon']):
            scenario_type = "medical"
        else:
            # Default to debt collection format
            scenario_type = "debt"
    
    print(f"Detected scenario type: {scenario_type}")
    
    # Call appropriate preprocessing function
    if scenario_type == "debt":
        return preprocess_debt_scenarios(csv_path, output_path, n_scenarios, seed, offset)
    elif scenario_type == "disaster":
        return preprocess_disaster_rescue_scenarios(csv_path, output_path, n_scenarios, seed, offset)
    elif scenario_type == "student":
        return preprocess_student_sleep_scenarios(csv_path, output_path, n_scenarios, seed, offset)
    elif scenario_type == "medical":
        return preprocess_medical_surgery_scenarios(csv_path, output_path, n_scenarios, seed, offset)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

# Add this new function to debug CSV issues
def debug_csv_structure(csv_path: str):
    """
    Debug function to understand CSV structure
    """
    print(f"\n=== Debugging CSV: {csv_path} ===")
    
    # Try different approaches
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        first_lines = [next(f) for _ in range(5)]
    
    print("First 5 lines of raw file:")
    for i, line in enumerate(first_lines):
        print(f"Line {i+1}: {repr(line[:200])}...")
    
    # Try reading with different separators
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(csv_path, sep=sep, nrows=3)
            print(f"\nWith separator '{sep}': {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
        except:
            print(f"\nSeparator '{sep}' failed")
    
    return


# Example usage:
if __name__ == "__main__":
    # Process debt collection CSV
    debt_scenarios = preprocess_all_scenarios(
        csv_path="data/debt_collection.csv",
        scenario_type="debt",
        output_path="config/scenarios_debt.json",
        n_scenarios=10
    )
    
    # Process disaster rescue CSV
    disaster_scenarios = preprocess_all_scenarios(
        csv_path="data/disaster_rescue.csv", 
        scenario_type="disaster",
        output_path="config/scenarios_disaster.json",
        n_scenarios=10
    )
    
    # Process student sleep CSV
    student_scenarios = preprocess_all_scenarios(
        csv_path="data/student_sleep.csv",
        scenario_type="student", 
        output_path="config/scenarios_student.json",
        n_scenarios=10
    )
    
    # Process medical surgery CSV
    medical_scenarios = preprocess_all_scenarios(
        csv_path="data/medical_surgery.csv",
        scenario_type="medical",
        output_path="config/scenarios_medical.json",
        n_scenarios=10
    )
    
    # Combine all scenarios
    all_scenarios = debt_scenarios + disaster_scenarios + student_scenarios + medical_scenarios
    with open("config/scenarios_all.json", 'w') as f:
        json.dump(all_scenarios, f, indent=2)
    
    print(f"\nProcessed {len(all_scenarios)} total scenarios across 4 domains")