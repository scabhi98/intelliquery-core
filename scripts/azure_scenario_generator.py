"""
Azure Log Analytics - Advanced Scenario Generator
Creates realistic incident scenarios with correlated events across time
Perfect for testing incident investigation and correlation queries
"""

import requests
import json
import datetime
import random
import time
import hashlib
import hmac
import base64
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
WORKSPACE_ID = 'your-workspace-id'
SHARED_KEY = 'your-shared-key'
LOG_TYPE = 'CustomApplicationLogs'

# ============================================================================
# Azure Log Analytics Client (same as before)
# ============================================================================
class AzureLogAnalytics:
    def __init__(self, workspace_id: str, shared_key: str):
        self.workspace_id = workspace_id
        self.shared_key = shared_key
        self.api_endpoint = f'https://{workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01'
    
    def build_signature(self, date: str, content_length: int, method: str, content_type: str, resource: str) -> str:
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
# Incident Scenario Generators
# ============================================================================

def generate_database_outage_scenario(base_time: datetime.datetime, duration_minutes: int = 30) -> List[Dict]:
    """
    Simulate a database outage with cascading failures
    """
    logs = []
    incident_id = f"INC-{random.randint(10000, 99999)}"
    
    # Initial warning signs (5 minutes before outage)
    warning_time = base_time - datetime.timedelta(minutes=5)
    for i in range(3):
        timestamp = warning_time + datetime.timedelta(seconds=i * 60)
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'DataProcessor',
            'Severity': 'Warning',
            'Message': 'Database connection pool exhaustion detected',
            'Category': 'Performance',
            'IncidentId': incident_id,
            'LogType': 'Trace'
        })
    
    # Outage begins - dependency failures
    outage_start = base_time
    for i in range(duration_minutes):
        timestamp = outage_start + datetime.timedelta(minutes=i)
        
        # Multiple dependency failures per minute
        for j in range(random.randint(5, 15)):
            logs.append({
                'TimeGenerated': (timestamp + datetime.timedelta(seconds=j * 4)).isoformat() + 'Z',
                'Application': random.choice(['WebApp', 'MobileAPI', 'DataProcessor']),
                'DependencyName': 'SQL-Database',
                'DependencyType': 'Database',
                'Target': 'sql-database.azure.com',
                'Duration': random.randint(30000, 60000),
                'Success': False,
                'ResultCode': '504',
                'IncidentId': incident_id,
                'LogType': 'Dependency'
            })
        
        # Failed requests
        for j in range(random.randint(10, 25)):
            logs.append({
                'TimeGenerated': (timestamp + datetime.timedelta(seconds=j * 2)).isoformat() + 'Z',
                'Application': random.choice(['WebApp', 'MobileAPI']),
                'RequestMethod': random.choice(['GET', 'POST']),
                'RequestPath': random.choice(['/api/users', '/api/orders', '/api/products']),
                'StatusCode': 503,
                'Duration': random.randint(5000, 30000),
                'Success': False,
                'IncidentId': incident_id,
                'LogType': 'Request'
            })
        
        # Exceptions
        for j in range(random.randint(3, 8)):
            logs.append({
                'TimeGenerated': (timestamp + datetime.timedelta(seconds=j * 7)).isoformat() + 'Z',
                'Application': random.choice(['WebApp', 'DataProcessor']),
                'ExceptionType': 'DatabaseConnectionException',
                'Message': 'DatabaseConnectionException: Unable to connect to database server',
                'StackTrace': 'at System.Data.SqlClient.SqlConnection.Open()\\n   at DataAccess.ExecuteQuery()',
                'Severity': 'Critical',
                'IncidentId': incident_id,
                'LogType': 'Exception'
            })
    
    # Resolution
    resolution_time = outage_start + datetime.timedelta(minutes=duration_minutes)
    logs.append({
        'TimeGenerated': resolution_time.isoformat() + 'Z',
        'Application': 'DataProcessor',
        'Severity': 'Information',
        'Message': 'Database connection restored',
        'Category': 'System',
        'IncidentId': incident_id,
        'LogType': 'Trace'
    })
    
    # Post-recovery monitoring
    for i in range(5):
        timestamp = resolution_time + datetime.timedelta(minutes=i + 1)
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'DataProcessor',
            'Severity': 'Information',
            'Message': 'System health check passed',
            'Category': 'System',
            'IncidentId': incident_id,
            'LogType': 'Trace'
        })
    
    return logs

def generate_security_incident_scenario(base_time: datetime.datetime) -> List[Dict]:
    """
    Simulate a security incident with multiple failed login attempts and eventual lockout
    """
    logs = []
    incident_id = f"SEC-{random.randint(10000, 99999)}"
    attacker_ip = f"{random.randint(100, 200)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    target_user = f"user_{random.randint(1000, 9999)}"
    
    # Multiple failed login attempts
    for i in range(25):
        timestamp = base_time + datetime.timedelta(seconds=i * 2)
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'AuthService',
            'RequestMethod': 'POST',
            'RequestPath': '/api/auth/login',
            'StatusCode': 401,
            'Duration': random.randint(100, 500),
            'ClientIP': attacker_ip,
            'Success': False,
            'UserId': target_user,
            'IncidentId': incident_id,
            'LogType': 'Request'
        })
        
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'AuthService',
            'Severity': 'Warning',
            'Message': f'Failed login attempt for user {target_user}',
            'Category': 'Security',
            'IncidentId': incident_id,
            'LogType': 'Trace'
        })
    
    # Account lockout
    lockout_time = base_time + datetime.timedelta(seconds=50)
    logs.append({
        'TimeGenerated': lockout_time.isoformat() + 'Z',
        'Application': 'AuthService',
        'EventName': 'AccountLocked',
        'UserId': target_user,
        'Properties': json.dumps({
            'reason': 'Multiple failed login attempts',
            'source_ip': attacker_ip,
            'attempt_count': 25
        }),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    logs.append({
        'TimeGenerated': lockout_time.isoformat() + 'Z',
        'Application': 'AuthService',
        'Severity': 'Critical',
        'Message': f'Account locked due to suspicious activity: {target_user}',
        'Category': 'Security',
        'IncidentId': incident_id,
        'LogType': 'Trace'
    })
    
    # IP blocking
    block_time = lockout_time + datetime.timedelta(seconds=10)
    logs.append({
        'TimeGenerated': block_time.isoformat() + 'Z',
        'Application': 'AuthService',
        'EventName': 'IPBlocked',
        'Properties': json.dumps({
            'blocked_ip': attacker_ip,
            'reason': 'Brute force attack detected',
            'duration': '24h'
        }),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    return logs

def generate_performance_degradation_scenario(base_time: datetime.datetime, duration_minutes: int = 45) -> List[Dict]:
    """
    Simulate gradual performance degradation leading to service slowdown
    """
    logs = []
    incident_id = f"PERF-{random.randint(10000, 99999)}"
    
    # Normal baseline
    for i in range(5):
        timestamp = base_time - datetime.timedelta(minutes=10 - i)
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'WebApp',
            'MetricName': 'CPU_Usage',
            'MetricValue': round(random.uniform(20, 35), 2),
            'Unit': 'Percent',
            'IncidentId': incident_id,
            'LogType': 'Performance'
        })
    
    # Gradual degradation
    for i in range(duration_minutes):
        timestamp = base_time + datetime.timedelta(minutes=i)
        
        # Increasing CPU
        cpu_usage = 40 + (i * 1.2)  # Gradually increases
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'WebApp',
            'MetricName': 'CPU_Usage',
            'MetricValue': min(round(cpu_usage, 2), 98),
            'Unit': 'Percent',
            'IncidentId': incident_id,
            'LogType': 'Performance'
        })
        
        # Slow requests
        if i > 10:  # After 10 minutes, requests start slowing
            for j in range(random.randint(5, 10)):
                logs.append({
                    'TimeGenerated': (timestamp + datetime.timedelta(seconds=j * 6)).isoformat() + 'Z',
                    'Application': 'WebApp',
                    'RequestMethod': 'GET',
                    'RequestPath': random.choice(['/api/products', '/api/search', '/api/dashboard']),
                    'StatusCode': 200,
                    'Duration': random.randint(5000, 15000),  # Slow but successful
                    'Success': True,
                    'IncidentId': incident_id,
                    'LogType': 'Request'
                })
        
        # Warnings
        if i > 15 and i % 5 == 0:
            logs.append({
                'TimeGenerated': timestamp.isoformat() + 'Z',
                'Application': 'WebApp',
                'Severity': 'Warning',
                'Message': f'High CPU usage detected: {min(round(cpu_usage, 2), 98)}%',
                'Category': 'Performance',
                'IncidentId': incident_id,
                'LogType': 'Trace'
            })
    
    return logs

def generate_deployment_rollback_scenario(base_time: datetime.datetime) -> List[Dict]:
    """
    Simulate a bad deployment that causes errors and gets rolled back
    """
    logs = []
    incident_id = f"DEPLOY-{random.randint(10000, 99999)}"
    version = f"v{random.randint(2, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    
    # Deployment start
    logs.append({
        'TimeGenerated': base_time.isoformat() + 'Z',
        'Application': 'PaymentGateway',
        'EventName': 'DeploymentStarted',
        'Properties': json.dumps({'version': version, 'environment': 'production'}),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    # Brief downtime during deployment
    deploy_time = base_time + datetime.timedelta(minutes=2)
    logs.append({
        'TimeGenerated': deploy_time.isoformat() + 'Z',
        'Application': 'PaymentGateway',
        'EventName': 'DeploymentCompleted',
        'Properties': json.dumps({'version': version, 'duration_seconds': 120}),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    # Errors start appearing
    error_start = deploy_time + datetime.timedelta(minutes=1)
    for i in range(30):
        timestamp = error_start + datetime.timedelta(seconds=i * 10)
        
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'PaymentGateway',
            'ExceptionType': 'InvalidOperationException',
            'Message': 'InvalidOperationException: Payment processor configuration invalid',
            'StackTrace': 'at Services.PaymentService.ProcessPayment()\\n   at Controllers.CheckoutController.CompleteOrder()',
            'Severity': 'Error',
            'IncidentId': incident_id,
            'LogType': 'Exception'
        })
        
        logs.append({
            'TimeGenerated': timestamp.isoformat() + 'Z',
            'Application': 'PaymentGateway',
            'RequestMethod': 'POST',
            'RequestPath': '/api/payments',
            'StatusCode': 500,
            'Duration': random.randint(1000, 3000),
            'Success': False,
            'IncidentId': incident_id,
            'LogType': 'Request'
        })
    
    # Rollback decision
    rollback_start = error_start + datetime.timedelta(minutes=5)
    logs.append({
        'TimeGenerated': rollback_start.isoformat() + 'Z',
        'Application': 'PaymentGateway',
        'EventName': 'RollbackStarted',
        'Properties': json.dumps({'reason': 'High error rate detected', 'target_version': 'v2.1.5'}),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    # Rollback completion
    rollback_complete = rollback_start + datetime.timedelta(minutes=2)
    logs.append({
        'TimeGenerated': rollback_complete.isoformat() + 'Z',
        'Application': 'PaymentGateway',
        'EventName': 'RollbackCompleted',
        'Properties': json.dumps({'version': 'v2.1.5', 'duration_seconds': 120}),
        'IncidentId': incident_id,
        'LogType': 'AppEvent'
    })
    
    # System stabilizes
    stable_time = rollback_complete + datetime.timedelta(minutes=1)
    logs.append({
        'TimeGenerated': stable_time.isoformat() + 'Z',
        'Application': 'PaymentGateway',
        'Severity': 'Information',
        'Message': 'Service health restored after rollback',
        'Category': 'System',
        'IncidentId': incident_id,
        'LogType': 'Trace'
    })
    
    return logs

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("Azure Log Analytics - Advanced Scenario Generator")
    print("=" * 70)
    
    if WORKSPACE_ID == 'your-workspace-id' or SHARED_KEY == 'your-shared-key':
        print("\n❌ ERROR: Please update WORKSPACE_ID and SHARED_KEY in the script!")
        return
    
    azure_logs = AzureLogAnalytics(WORKSPACE_ID, SHARED_KEY)
    
    print(f"\n📊 Configuration:")
    print(f"   Workspace ID: {WORKSPACE_ID}")
    print(f"   Log Type: {LOG_TYPE}")
    
    # Define scenarios with timestamps spread over 3 months
    now = datetime.datetime.utcnow()
    scenarios = [
        ("Database Outage", generate_database_outage_scenario(now - datetime.timedelta(days=15), 30)),
        ("Security Incident", generate_security_incident_scenario(now - datetime.timedelta(days=45))),
        ("Performance Degradation", generate_performance_degradation_scenario(now - datetime.timedelta(days=7), 45)),
        ("Deployment Rollback", generate_deployment_rollback_scenario(now - datetime.timedelta(days=30))),
        ("Recent Database Issue", generate_database_outage_scenario(now - datetime.timedelta(days=2), 15)),
    ]
    
    print(f"\n🎬 Generating {len(scenarios)} incident scenarios...")
    total_logs = 0
    
    for scenario_name, logs in scenarios:
        print(f"\n📋 Scenario: {scenario_name}")
        print(f"   Logs: {len(logs)}")
        print(f"   Time Range: {logs[0]['TimeGenerated']} to {logs[-1]['TimeGenerated']}")
        
        # Send in chunks to avoid payload size limits
        chunk_size = 100
        for i in range(0, len(logs), chunk_size):
            chunk = logs[i:i+chunk_size]
            status_code = azure_logs.post_data(chunk, LOG_TYPE)
            
            if status_code == 200:
                print(f"   ✅ Sent chunk {i//chunk_size + 1} ({len(chunk)} logs)")
            else:
                print(f"   ❌ Failed chunk {i//chunk_size + 1} (Status: {status_code})")
        
        total_logs += len(logs)
        time.sleep(2)  # Brief delay between scenarios
    
    print(f"\n✅ Completed! Total logs sent: {total_logs}")
    print(f"\n📝 Investigation Queries:")
    print(f"""
   // Find all incidents
   {LOG_TYPE}_CL 
   | where isnotempty(IncidentId_s)
   | summarize 
       StartTime=min(TimeGenerated), 
       EndTime=max(TimeGenerated),
       LogCount=count(),
       LogTypes=make_set(LogType_s)
     by IncidentId_s
   | order by StartTime desc
   
   // Investigate specific incident
   {LOG_TYPE}_CL 
   | where IncidentId_s == "INC-12345"  // Replace with actual ID
   | order by TimeGenerated asc
   | project TimeGenerated, Application_s, LogType_s, Message_s, Severity_s
   
   // Timeline of a database outage
   {LOG_TYPE}_CL 
   | where IncidentId_s startswith "INC-"
   | summarize count() by bin(TimeGenerated, 1m), LogType_s
   | render timechart
    """)

if __name__ == "__main__":
    main()
