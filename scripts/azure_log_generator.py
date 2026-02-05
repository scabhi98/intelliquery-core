"""
Azure Log Analytics - Realistic Telemetry Generator
Pushes random application telemetry data to Azure Log Analytics workspace
Generates logs distributed over a 3-month period for realistic investigation scenarios
"""

import requests
import json
import datetime
import random
import time
import hashlib
import hmac
import base64
from typing import List, Dict

# ============================================================================
# CONFIGURATION - Update these values with your workspace details
# ============================================================================
WORKSPACE_ID = 'your-workspace-id'  # Your Log Analytics Workspace ID
SHARED_KEY = 'your-shared-key'      # Your Primary or Secondary Key
LOG_TYPE = 'CustomApplicationLogs'   # Custom log type name

# Time distribution settings
MONTHS_OF_DATA = 3  # Generate logs spanning this many months back
BUSINESS_HOURS_WEIGHT = 0.7  # 70% of logs during business hours (9 AM - 6 PM)
WEEKDAY_WEIGHT = 0.8  # 80% of logs on weekdays vs weekends

# ============================================================================
# Azure Log Analytics API Configuration
# ============================================================================
class AzureLogAnalytics:
    def __init__(self, workspace_id: str, shared_key: str):
        self.workspace_id = workspace_id
        self.shared_key = shared_key
        self.api_endpoint = f'https://{workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01'
    
    def build_signature(self, date: str, content_length: int, method: str, content_type: str, resource: str) -> str:
        """Build the authorization signature for Azure Log Analytics"""
        x_headers = f'x-ms-date:{date}'
        string_to_hash = f'{method}\n{content_length}\n{content_type}\n{x_headers}\n{resource}'
        bytes_to_hash = bytes(string_to_hash, encoding="utf-8")
        decoded_key = base64.b64decode(self.shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode()
        authorization = f'SharedKey {self.workspace_id}:{encoded_hash}'
        return authorization
    
    def post_data(self, body: List[Dict], log_type: str) -> int:
        """Post data to Azure Log Analytics"""
        body_json = json.dumps(body)
        body_bytes = body_json.encode('utf-8')
        content_length = len(body_bytes)
        
        rfc1123date = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        content_type = 'application/json'
        signature = self.build_signature(rfc1123date, content_length, 'POST', content_type, '/api/logs')
        
        headers = {
            'content-type': content_type,
            'Authorization': signature,
            'Log-Type': log_type,
            'x-ms-date': rfc1123date
        }
        
        response = requests.post(self.api_endpoint, data=body_bytes, headers=headers)
        return response.status_code

# ============================================================================
# Timestamp Generation
# ============================================================================

def generate_realistic_timestamp(base_time: datetime.datetime = None) -> str:
    """
    Generate a realistic timestamp distributed over the past MONTHS_OF_DATA months.
    Weights timestamps toward business hours and weekdays for more realistic patterns.
    """
    if base_time is None:
        base_time = datetime.datetime.utcnow()
    
    # Calculate random time within the past N months
    days_back = random.randint(0, MONTHS_OF_DATA * 30)
    random_date = base_time - datetime.timedelta(days=days_back)
    
    # Decide if this should be a business hour timestamp (70% chance)
    is_business_hours = random.random() < BUSINESS_HOURS_WEIGHT
    
    if is_business_hours:
        # Business hours: 9 AM - 6 PM
        hour = random.randint(9, 17)
    else:
        # Off hours: weighted toward early morning/late night
        hour = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23])
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    microsecond = random.randint(0, 999999)
    
    # Create timestamp
    timestamp = random_date.replace(
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond
    )
    
    # Apply weekday weighting (80% weekdays, 20% weekends)
    if timestamp.weekday() >= 5:  # Saturday or Sunday
        if random.random() > (1 - WEEKDAY_WEIGHT):
            # Shift to a weekday
            days_to_shift = random.randint(1, 2)
            timestamp = timestamp - datetime.timedelta(days=days_to_shift)
    
    return timestamp.isoformat() + 'Z'

def generate_correlated_timestamp(base_timestamp: str, max_seconds_diff: int = 300) -> str:
    """
    Generate a timestamp close to a base timestamp (for correlated events).
    Useful for exceptions that follow failed requests, etc.
    """
    base_dt = datetime.datetime.fromisoformat(base_timestamp.replace('Z', ''))
    seconds_diff = random.randint(-max_seconds_diff, max_seconds_diff)
    new_dt = base_dt + datetime.timedelta(seconds=seconds_diff)
    return new_dt.isoformat() + 'Z'

# ============================================================================
# Sample Data Generators
# ============================================================================

# Sample application data
APP_NAMES = ['WebApp', 'MobileAPI', 'DataProcessor', 'AuthService', 'PaymentGateway']
ENDPOINTS = [
    '/api/users', '/api/products', '/api/orders', '/api/payments', '/api/auth',
    '/api/search', '/api/cart', '/api/checkout', '/api/profile', '/api/dashboard'
]
HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
    'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36'
]
EXCEPTION_TYPES = [
    'NullReferenceException', 'TimeoutException', 'DatabaseConnectionException',
    'ValidationException', 'AuthenticationException', 'RateLimitException',
    'ServiceUnavailableException', 'InvalidOperationException'
]
DEPENDENCY_NAMES = [
    'SQL-Database', 'Redis-Cache', 'Azure-Storage', 'External-API',
    'Message-Queue', 'CosmosDB', 'MongoDB', 'Elasticsearch'
]

def generate_trace_log() -> Dict:
    """Generate a trace/diagnostic log entry"""
    severity_levels = ['Verbose', 'Information', 'Warning', 'Error', 'Critical']
    severity = random.choice(severity_levels)
    
    messages = {
        'Verbose': [
            'Entering method ProcessRequest',
            'Cache lookup performed',
            'Query executed successfully',
            'Request validation started'
        ],
        'Information': [
            'User login successful',
            'Data processing completed',
            'Cache refreshed',
            'Configuration loaded'
        ],
        'Warning': [
            'High memory usage detected',
            'Slow query execution time',
            'Cache miss rate above threshold',
            'Retry attempt initiated'
        ],
        'Error': [
            'Failed to connect to database',
            'API call timeout',
            'Invalid input data received',
            'Resource not found'
        ],
        'Critical': [
            'Service unavailable',
            'Data corruption detected',
            'Security breach attempt',
            'System out of memory'
        ]
    }
    
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'Severity': severity,
        'Message': random.choice(messages[severity]),
        'Category': random.choice(['System', 'Security', 'Application', 'Performance']),
        'LogType': 'Trace'
    }

def generate_app_event() -> Dict:
    """Generate an application event log"""
    event_types = [
        'UserLogin', 'UserLogout', 'PageView', 'ButtonClick', 
        'FormSubmit', 'SearchQuery', 'FileUpload', 'EmailSent'
    ]
    
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'EventName': random.choice(event_types),
        'UserId': f'user_{random.randint(1000, 9999)}',
        'SessionId': f'session_{random.randint(100000, 999999)}',
        'Properties': json.dumps({
            'page': random.choice(['/home', '/products', '/cart', '/checkout', '/profile']),
            'action': random.choice(['view', 'click', 'submit', 'cancel']),
            'duration_ms': random.randint(50, 5000)
        }),
        'LogType': 'AppEvent'
    }

def generate_dependency_log() -> Dict:
    """Generate a dependency call log"""
    dependency = random.choice(DEPENDENCY_NAMES)
    success = random.random() > 0.1  # 90% success rate
    duration = random.randint(10, 2000) if success else random.randint(2000, 30000)
    
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'DependencyName': dependency,
        'DependencyType': dependency.split('-')[1],
        'Target': f'{dependency.lower()}.{random.choice(["azure.com", "internal", "cloud.provider"])}',
        'Duration': duration,
        'Success': success,
        'ResultCode': '200' if success else random.choice(['500', '503', '504', '408']),
        'LogType': 'Dependency'
    }

def generate_request_log() -> Dict:
    """Generate an HTTP request log"""
    method = random.choice(HTTP_METHODS)
    endpoint = random.choice(ENDPOINTS)
    success = random.random() > 0.05  # 95% success rate
    
    status_code = 200 if success else random.choice([400, 401, 403, 404, 500, 502, 503])
    duration = random.randint(50, 2000) if success else random.randint(2000, 10000)
    
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'RequestMethod': method,
        'RequestPath': endpoint,
        'StatusCode': status_code,
        'Duration': duration,
        'UserAgent': random.choice(USER_AGENTS),
        'ClientIP': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
        'Success': success,
        'LogType': 'Request'
    }

def generate_exception_log() -> Dict:
    """Generate an exception/error log"""
    exception_type = random.choice(EXCEPTION_TYPES)
    
    stack_traces = [
        'at System.Data.SqlClient.SqlConnection.Open()\n   at DataAccess.ExecuteQuery()',
        'at API.Controllers.UserController.GetUser()\n   at Microsoft.AspNetCore.Mvc.Infrastructure.ActionInvoker.Invoke()',
        'at Services.PaymentService.ProcessPayment()\n   at Controllers.CheckoutController.CompleteOrder()',
        'at Auth.Middleware.ValidateToken()\n   at Microsoft.AspNetCore.Authentication.AuthenticationMiddleware.Invoke()'
    ]
    
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'ExceptionType': exception_type,
        'Message': f'{exception_type}: An error occurred during processing',
        'StackTrace': random.choice(stack_traces),
        'Severity': random.choice(['Error', 'Critical']),
        'InnerException': random.choice([None, 'TimeoutException', 'ArgumentNullException']),
        'LogType': 'Exception'
    }

def generate_performance_log() -> Dict:
    """Generate a performance metric log"""
    return {
        'TimeGenerated': generate_realistic_timestamp(),
        'Application': random.choice(APP_NAMES),
        'MetricName': random.choice(['CPU_Usage', 'Memory_Usage', 'Request_Rate', 'Error_Rate']),
        'MetricValue': round(random.uniform(0, 100), 2),
        'Unit': random.choice(['Percent', 'MB', 'Count/sec', 'Milliseconds']),
        'LogType': 'Performance'
    }

# ============================================================================
# Main Execution
# ============================================================================

def generate_batch_logs(count: int = 10) -> List[Dict]:
    """Generate a batch of mixed log entries"""
    logs = []
    
    # Distribution of log types (weighted)
    log_generators = [
        (generate_trace_log, 30),      # 30% traces
        (generate_request_log, 25),    # 25% requests
        (generate_dependency_log, 20), # 20% dependencies
        (generate_app_event, 15),      # 15% app events
        (generate_performance_log, 7), # 7% performance
        (generate_exception_log, 3)    # 3% exceptions
    ]
    
    # Create weighted list
    weighted_generators = []
    for generator, weight in log_generators:
        weighted_generators.extend([generator] * weight)
    
    for _ in range(count):
        generator = random.choice(weighted_generators)
        logs.append(generator())
    
    return logs

def main():
    """Main execution function"""
    print("=" * 70)
    print("Azure Log Analytics - Telemetry Data Generator")
    print("=" * 70)
    
    # Validate configuration
    if WORKSPACE_ID == 'your-workspace-id' or SHARED_KEY == 'your-shared-key':
        print("\n❌ ERROR: Please update WORKSPACE_ID and SHARED_KEY in the script!")
        print("\nTo find these values:")
        print("1. Go to Azure Portal")
        print("2. Navigate to your Log Analytics Workspace")
        print("3. Go to 'Agents' under 'Settings'")
        print("4. Copy the Workspace ID and Primary Key")
        return
    
    # Initialize Azure Log Analytics client
    azure_logs = AzureLogAnalytics(WORKSPACE_ID, SHARED_KEY)
    
    # Calculate time range
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=MONTHS_OF_DATA * 30)
    
    print(f"\n📊 Configuration:")
    print(f"   Workspace ID: {WORKSPACE_ID}")
    print(f"   Log Type: {LOG_TYPE}")
    print(f"   Time Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: {MONTHS_OF_DATA} months of historical data")
    print(f"\n🚀 Starting log generation...")
    
    batch_num = 1
    total_logs_sent = 0
    
    try:
        # Generate and send logs in batches
        for i in range(5):  # Send 5 batches
            print(f"\n📦 Generating batch {batch_num}...")
            
            # Generate mixed logs
            logs = generate_batch_logs(count=20)
            
            # Show log type distribution
            log_types = {}
            for log in logs:
                log_type = log['LogType']
                log_types[log_type] = log_types.get(log_type, 0) + 1
            
            print(f"   Generated {len(logs)} logs: {log_types}")
            
            # Send to Azure
            print(f"   Sending to Azure Log Analytics...")
            status_code = azure_logs.post_data(logs, LOG_TYPE)
            
            if status_code == 200:
                print(f"   ✅ Successfully sent batch {batch_num} (Status: {status_code})")
                total_logs_sent += len(logs)
            else:
                print(f"   ❌ Failed to send batch {batch_num} (Status: {status_code})")
            
            batch_num += 1
            
            # Wait before sending next batch
            if i < 4:  # Don't wait after last batch
                wait_time = random.randint(2, 5)
                print(f"   ⏳ Waiting {wait_time} seconds before next batch...")
                time.sleep(wait_time)
        
        print(f"\n✅ Completed! Total logs sent: {total_logs_sent}")
        print(f"\n📝 Sample KQL queries to explore your data:")
        print(f"   // View all logs")
        print(f"   {LOG_TYPE}_CL | take 100")
        print(f"   ")
        print(f"   // Time distribution")
        print(f"   {LOG_TYPE}_CL | summarize count() by bin(TimeGenerated, 1d) | render timechart")
        print(f"   ")
        print(f"   // Logs by type")
        print(f"   {LOG_TYPE}_CL | summarize count() by LogType_s | render piechart")
        print(f"   ")
        print(f"   // Recent exceptions")
        print(f"   {LOG_TYPE}_CL | where LogType_s == 'Exception' | order by TimeGenerated desc")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
