import os
import sys
import json
import asyncio
import logging
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from enum import Enum
from email_tools import list_unread_emails, summarize_email

import psutil
import docker
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from dotenv import load_dotenv
from imap_tools import MailBox, AND
import schedule
import yaml

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END,START

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("devops_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Alert:
    timestamp: datetime
    level: Priority
    source: str
    message: str
    resolved: bool = False

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    processes: int
    uptime: float

class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    alerts: List[Alert]
    metrics: Optional[SystemMetrics]

class DevOpsAssistant:
    def __init__(self):
        self.config = self._load_config()
        self.alerts: List[Alert] = []
        self.monitoring_active = False
        self.docker_client = None

        # Initialize LLM
        self.raw_llm = init_chat_model(
            self.config['chat_model'],
            model_provider='ollama'
        )
        

        # Setup monitoring thresholds
        self.thresholds = {
            'cpu': self.config.get('cpu_threshold', 80),
            'memory': self.config.get('memory_threshold', 85),
            'disk': self.config.get('disk_threshold', 90)
        }

        # Initialize Docker client if available
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and config file"""
        config = {
            'chat_model': os.getenv('CHAT_MODEL', 'qwen3:8b'),
            'imap_host': os.getenv('IMAP_HOST'),
            'imap_user': os.getenv('IMAP_USER'),
            'imap_password': os.getenv('IMAP_PASSWORD'),
            'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', '30')),
            'alert_webhook': os.getenv('ALERT_WEBHOOK'),
            'cpu_threshold': float(os.getenv('CPU_THRESHOLD', '80')),
            'memory_threshold': float(os.getenv('MEMORY_THRESHOLD', '85')),
            'disk_threshold': float(os.getenv('DISK_THRESHOLD', '90')),
        }

        # Load additional config from file if exists
        config_file = Path('devops_config.yaml')
        if config_file.exists():
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)

        return config

    def add_alert(self, level: Priority, source: str, message: str):
        """Add a new alert to the system"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            source=source,
            message=message
        )
        self.alerts.append(alert)
        logger.warning(f"Alert [{level.name}] from {source}: {message}")

        # Send webhook notification if configured
        if self.config.get('alert_webhook'):
            self._send_webhook_alert(alert)

    def _send_webhook_alert(self, alert: Alert):
        """Send alert to configured webhook"""
        try:
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.name,
                'source': alert.source,
                'message': alert.message
            }
            requests.post(self.config['alert_webhook'], json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

# Initialize the assistant
assistant = DevOpsAssistant()

# Enhanced Tools
@tool
def get_system_metrics() -> str:
    """Get comprehensive system metrics including CPU, memory, disk, network, and processes."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        # Network metrics
        network_io = psutil.net_io_counters()

        # Process metrics
        processes = len(psutil.pids())
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # Check thresholds and create alerts
        if cpu_percent > assistant.thresholds['cpu']:
            assistant.add_alert(Priority.HIGH, "CPU", f"CPU usage at {cpu_percent}%")
        if memory.percent > assistant.thresholds['memory']:
            assistant.add_alert(Priority.HIGH, "Memory", f"Memory usage at {memory.percent}%")
        if disk_usage.percent > assistant.thresholds['disk']:
            assistant.add_alert(Priority.CRITICAL, "Disk", f"Disk usage at {disk_usage.percent}%")
            
        metrics = {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency": cpu_freq.current if cpu_freq else "N/A"
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2)
            },
            "swap": {
                "percent": swap.percent,
                "used_gb": round(swap.used / (1024**3), 2),
                "total_gb": round(swap.total / (1024**3), 2)
            },
            "disk": {
                "percent": disk_usage.percent,
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2)
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "system": {
                "processes": processes,
                "uptime_hours": round(uptime / 3600, 2)
            }
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error getting system metrics: {e}"

@tool
def monitor_docker_containers():
    """
    Monitors active Docker containers and reports their CPU and memory usage.
    Handles missing 'system_cpu_usage' from newer Docker Engine APIs.
    """
    try:
        containers = assistant.docker_client.containers.list()
        if not containers:
            return "🐳 No running Docker containers found."

        output = []
        for container in containers:
            stats = container.stats(stream=False)
            name = container.name
            status = container.status
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats'].get('limit', 1)
            mem_percent = (mem_usage / mem_limit) * 100 if mem_limit else 0

            cpu_percent = 0.0
            try:
                cpu_stats = stats['cpu_stats']
                precpu_stats = stats['precpu_stats']

                # Newer Docker versions don't include 'system_cpu_usage'
                cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)

                if cpu_delta > 0 and system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage'].get('percpu_usage', [])) * 100
            except Exception:
                cpu_percent = 0.0  # fail-safe

            output.append(f"🟢 {name} | Status: {status} | CPU: {cpu_percent:.2f}% | Memory: {mem_percent:.2f}%")

        return "\n".join(output)
    except Exception as e:
        return f"❌ Error monitoring Docker containers: {e}"


@tool
def advanced_git_analysis() -> str:
    """Perform advanced Git repository analysis including branch status, commit trends, and code quality metrics."""
    try:
        analysis = {}

        # Basic git status
        status_result = subprocess.run(['git', 'status', '--porcelain'],
                                       capture_output=True, text=True, check=True)
        analysis['uncommitted_changes'] = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0

        # Branch information
        branch_result = subprocess.run(['git', 'branch', '-r'],
                                       capture_output=True, text=True, check=True)
        analysis['remote_branches'] = len([b for b in branch_result.stdout.split('\n') if b.strip()])

        # Commit statistics (last 30 days)
        since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        commits_result = subprocess.run(['git', 'log', '--since', since_date, '--oneline'],
                                        capture_output=True, text=True, check=True)
        analysis['commits_last_30_days'] = len(commits_result.stdout.strip().split('\n')) if commits_result.stdout.strip() else 0

        # Contributors
        contributors_result = subprocess.run(['git', 'shortlog', '-sn', '--since', since_date],
                                            capture_output=True, text=True, check=True)
        analysis['active_contributors'] = len(contributors_result.stdout.strip().split('\n')) if contributors_result.stdout.strip() else 0

        # Repository size
        size_result = subprocess.run(['git', 'count-objects', '-vH'],
                                     capture_output=True, text=True, check=True)
        analysis['repo_info'] = size_result.stdout

        # Check for large files
        large_files_result = subprocess.run(['find', '.', '-size', '+10M', '-type', 'f'],
                                           capture_output=True, text=True)
        large_files = [f for f in large_files_result.stdout.split('\n') if f.strip() and not f.startswith('./.git/')]
        analysis['large_files_count'] = len(large_files)

        if large_files:
            assistant.add_alert(Priority.MEDIUM, "Git", f"Found {len(large_files)} large files in repository")

        return json.dumps(analysis, indent=2)
    except subprocess.CalledProcessError as e:
        return f"Git analysis error: {e}"
    except Exception as e:
        return f"Error performing git analysis: {e}"

@tool
def security_audit() -> str:
    """Perform a basic security audit of the system and codebase."""
    audit_results = {}
    try:
        # Check for common security issues in Python files
        security_issues = []

        for root, dirs, files in os.walk('.'):
            # Skip .git and other hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Check for common security anti-patterns
                        if 'eval(' in content:
                            security_issues.append(f"{file_path}: Uses eval() - potential code injection")
                        if 'exec(' in content:
                            security_issues.append(f"{file_path}: Uses exec() - potential code injection")
                        if 'shell=True' in content:
                            security_issues.append(f"{file_path}: Uses shell=True - potential command injection")
                        if 'password' in content.lower() and '=' in content:
                            security_issues.append(f"{file_path}: Potential hardcoded password")
                    except Exception:
                        continue

        audit_results['security_issues'] = security_issues

        # Check file permissions
        sensitive_files = ['.env', 'config.yaml', 'devops.config.yaml']
        permission_issues = []

        for file in sensitive_files:
            if os.path.exists(file):
                stat_info = os.stat(file)
                permissions = oct(stat_info.st_mode)[-3:]
                if permissions != '600':
                    permission_issues.append(f"{file}: Permissions {permissions} (should be 600)")

        audit_results['permission_issues'] = permission_issues

        # Check for dependency vulnerabilities (if requirements.txt exists)
        if os.path.exists("requirements.txt"):
            try:
                pip_audit = subprocess.run(['pip-audit', '--format', 'json'],
                                          capture_output=True, text=True, timeout=30)
                if pip_audit.returncode == 0:
                    audit_results['dependency_audit'] = json.loads(pip_audit.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                audit_results['dependency_audit'] = "pip-audit not available or timed out"

        # Create alerts for critical issues
        critical_issues = len([issue for issue in security_issues if 'injection' in issue])
        if critical_issues > 0:
            assistant.add_alert(Priority.CRITICAL, "Security", f"Found {critical_issues} potential injection vulnerabilities")

        return json.dumps(audit_results, indent=2)
    except Exception as e:
        return f"Error performing security audit: {e}"

@tool
def automated_deployment() -> str:
    """Simulate automated deployment with pre-deployment checks."""
    deployment_steps = []
    try:
        # Pre-deployment checks
        deployment_steps.append("Starting pre-deployment checks...")

        # Check git status
        git_result = subprocess.run(['git', 'status', '--porcelain'],
                                   capture_output=True, text=True)
        if git_result.stdout.strip():
            deployment_steps.append("Uncommitted changes detected")
            return '\n'.join(deployment_steps)
        deployment_steps.append("Git repository is clean")

        # Run tests
        test_result = subprocess.run(['python', '-m', 'pytest', '--maxfail=1'],
                                    capture_output=True, text=True, timeout=60)
        if test_result.returncode != 0:
            deployment_steps.append("Tests failed")
            deployment_steps.append(f"Test output: {test_result.stdout}")
            return '\n'.join(deployment_steps)
        deployment_steps.append("All tests passed")

        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > 80 or memory_percent > 80:
            deployment_steps.append("High system resource usage detected")
            assistant.add_alert(Priority.MEDIUM, "Deployment", "High resource usage during deployment")
        deployment_steps.append("System resources check passed")

        # Simulate deployment steps
        deployment_steps.append("Starting deployment...")
        deployment_steps.append("Building application...")
        time.sleep(1)  # Simulate build time
        deployment_steps.append("Build completed successfully")

        deployment_steps.append("Deploying to staging...")
        time.sleep(1)  # Simulate deployment time
        deployment_steps.append("Staging deployment completed")

        deployment_steps.append("Running smoke tests...")
        time.sleep(1)  # Simulate test time
        deployment_steps.append("Smoke tests passed")

        deployment_steps.append("Deployment completed successfully")

        return '\n'.join(deployment_steps)
    except subprocess.TimeoutExpired:
        deployment_steps.append("Deployment timed out")
        return '\n'.join(deployment_steps)
    except Exception as e:
        deployment_steps.append(f"Deployment failed: {e}")
        return '\n'.join(deployment_steps)

@tool
def generate_project_report() -> str:
    """Generate a comprehensive project health report."""
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_name": os.path.basename(os.getcwd()),
            "git_analysis": {},
            "system_metrics": {},
            "alerts_summary": {},
            "recommendations": []
        }

        # Git analysis
        try:
            git_log = subprocess.run(['git', 'log', '--oneline', '-10'],
                                    capture_output=True, text=True, check=True)
            report['git_analysis']['recent_commits'] = len(git_log.stdout.strip().split('\n'))
            git_status = subprocess.run(['git', 'status', '--porcelain'],
                                      capture_output=True, text=True, check=True)
            report['git_analysis']['uncommitted_files'] = len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
        except:
            report['git_analysis']['error'] = 'Git not available or not a git repository'

        # System metrics
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        report["system_metrics"] = {
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2)
        }

        # Alerts summary
        recent_alerts = [alert for alert in assistant.alerts
                         if alert.timestamp > datetime.now() - timedelta(hours=24)]

        report["alerts_summary"] = {
            "total_alerts_24h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.level == Priority.CRITICAL]),
            "high_alerts": len([a for a in recent_alerts if a.level == Priority.HIGH]),
            "unresolved_alerts": len([a for a in recent_alerts if not a.resolved])
        }
        
        if cpu > 80:
            report["recommendations"].append("Consider optimizing CPU-intensive processes")
        if memory.percent > 85:
            report["recommendations"].append("Memory usage is high - consider adding more RAM or optimizing memory usage")
        if disk.percent > 90:
            report["recommendations"].append("Disk space is critically low - cleanup required")
        if report["git_analysis"].get("uncommitted_files", 0) > 0:
            report["recommendations"].append("Commit or stash uncommitted changes")

        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error generating project report: {e}"

@tool
def list_unread_emails(limit: int = 5) -> str:
    """Use IMAP to list unread emails from your Gmail inbox. Returns UID, date, sender, subject."""
    host = assistant.config.get('imap_host')
    user = assistant.config.get('imap_user')
    pw = assistant.config.get('imap_password')

    if not all([host, user, pw]):
        return ("❌ Email configuration not complete. Set these environment variables:\n"
                "IMAP_HOST=imap.gmail.com\nIMAP_USER=your_email@gmail.com\nIMAP_PASSWORD=your_app_password")

    try:
        with MailBox(host, port=993, timeout=30).login(user, pw, initial_folder='INBOX') as mb:
            messages = list(mb.fetch(
                criteria=AND(seen=False),
                headers_only=True,
                mark_seen=False,
                reverse=True,
                limit=limit
            ))
            if not messages:
                return "📭 No unread messages found."

            lines = [f"📧 Found {len(messages)} unread emails:\n"]
            for msg in messages:
                dt = msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else "Unknown date"
                sender = str(msg.from_)[:30]
                subject = msg.subject[:50] if msg.subject else "(no subject)"
                lines.append(f"   UID {msg.uid} | {dt}")
                lines.append(f"   From: {sender}")
                lines.append(f"   Subject: {subject}")
                lines.append("")
            return "\n".join(lines)

    except Exception as e:
        return f"❌ Error connecting to Gmail: {e}"


@tool
def summarize_email(uid: str) -> str:
    """Given a UID, this tool connects to IMAP and summarizes the full email content."""
    
    host = assistant.config.get('imap_host')
    user = assistant.config.get('imap_user')
    pw = assistant.config.get('imap_password')

    if not all([host, user, pw]):
        return "❌ Email config incomplete. Check your .env file."
    if not uid or not uid.isdigit():
        return "❌ Invalid UID. Use 'list_unread_emails' to find one."

    try:
        with MailBox(host, port=993, timeout=30).login(user, pw, initial_folder='INBOX') as mb:
            messages = list(mb.fetch(AND(uid=uid), mark_seen=False, limit=1))
            if not messages:
                return f"❌ No email found with UID {uid}."

            mail = messages[0]
            body = mail.text or re.sub('<[^<]+?>', '', mail.html) if mail.html else "No content."

            prompt = f"""
Summarize this email in 2-3 clear sentences:

From: {mail.from_}
Subject: {mail.subject}
Date: {mail.date.strftime("%Y-%m-%d %H:%M")}

Content:
{body[:500]}...

Focus on the purpose and any action items."""
            response = assistant.raw_llm.invoke([HumanMessage(content=prompt)])
            return f"📧 Summary:\nFrom: {mail.from_}\nSubject: {mail.subject}\nDate: {mail.date}\n\n{response.content}"

    except Exception as e:
        return f"❌ Error summarizing email: {str(e)}"




@tool
def check_git_status() -> str:
    """Run 'git status' and return the result with additional analysis."""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True)
        status_output = result.stdout

        # Additional git info
        branch_result = subprocess.run(['git', 'branch', '--show-current'],
                                      capture_output=True, text=True, check=True)
        current_branch = branch_result.stdout.strip()

        try:
            ahead_behind = subprocess.run(['git', 'status', '-b', '--porcelain'],
                                         capture_output=True, text=True, check=True)
            branch_info = ahead_behind.stdout.split('\n')[0] if ahead_behind.stdout else ""
        except:
            branch_info = ""
        enhanced_status = f"Current branch: {current_branch}\n"
        if branch_info:
            enhanced_status += f"Branch Status: {branch_info}\n"
        enhanced_status += f"\n{status_output}"
        return enhanced_status
    except subprocess.CalledProcessError as e:
        return f"Error running git status: {e}"

@tool
def list_recent_commits(n: int = 5) -> str:
    """List the most recent n git commits with enhanced information."""
    try:
        cmd = ['git', 'log', f'--n={n}', '--pretty=format:%h - %s (%an, %ar) [%d]', '--graph']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f'Error fetching commits: {e}'

@tool
def run_tests() -> str:
    """Fixed test runner that excludes venv and only runs actual project tests."""
    try:
        # Find only project test files (exclude venv, site-packages)
        test_files = []
        exclude_dirs = {'venv', 'site-packages', '__pycache__', '.git', 'node_modules'}
        
        for root, dirs, files in os.walk('.'):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if ((file.startswith('test_') or file.endswith('_test.py')) and 
                    file.endswith('.py') and 
                    not any(ex in root for ex in exclude_dirs)):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            return "❌ No project test files found. Create test_*.py files in your project directory."
        
        # Validate test files have actual test functions
        valid_test_files = []
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if file contains test functions
                    if 'def test_' in content or 'class Test' in content:
                        valid_test_files.append(test_file)
                    else:
                        logger.warning(f"Test file {test_file} contains no test functions")
            except Exception as e:
                logger.warning(f"Could not read test file {test_file}: {e}")
        
        if not valid_test_files:
            return "❌ No valid test files found. Test files must contain functions starting with 'test_' or test classes."
        
        # Run pytest with verbose output and discovery
        cmd = [sys.executable, '-m', 'pytest', '-v', '--tb=short'] + valid_test_files[:5]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        output = f"🔍 Found {len(test_files)} test files, {len(valid_test_files)} valid\n"
        output += f"🧪 Running tests on: {', '.join([Path(f).name for f in valid_test_files[:5]])}\n\n"
        
        if result.returncode == 0:
            output += "✅ All tests passed!\n"
            output += result.stdout[-300:] if result.stdout else ""
        else:
            output += "❌ Tests failed:\n"
            # Show both stdout and stderr for better debugging
            if result.stdout:
                output += "STDOUT:\n" + result.stdout[-400:] + "\n"
            if result.stderr:
                output += "STDERR:\n" + result.stderr[-400:] + "\n"
                
        return output
        
    except subprocess.TimeoutExpired:
        return "❌ Tests timed out after 60 seconds"
    except Exception as e:
        return f"❌ Test execution error: {e}"


@tool
def explain_traceback(log: str) -> str:
    """Provide a detailed explanation of a Python traceback log with suggested fixes."""
    prompt = f"""
Analyze this Python traceback and provide:
1. A clear explanation of what went wrong
2. The root cause of the error
3. Specific steps to fix the issue
4. Best practices to prevent similar errors

Traceback:
{log}
"""
    return assistant.raw_llm.invoke(prompt).content

@tool
def get_active_alerts() -> str:
    """Get all active (unresolved) alerts from the system."""
    active_alerts = [alert for alert in assistant.alerts if not alert.resolved]

    if not active_alerts:
        return "No active alerts."

    alert_data = []
    for alert in active_alerts:
        alert_data.append({
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.name,
            "source": alert.source,
            "message": alert.message
        })
    return json.dumps(alert_data, indent=2)

# Create tool list
tools = [
    get_system_metrics,
    monitor_docker_containers,
    advanced_git_analysis,
    security_audit,
    automated_deployment,
    generate_project_report,
    list_unread_emails,
    summarize_email,
    check_git_status,
    list_recent_commits,
    run_tests,
    explain_traceback,
    get_active_alerts
]

# Agent Graph Setup
def create_agent_graph():
    """Create and configure the agent graph"""
    # Bind tools to LLM
    llm_with_tools = assistant.raw_llm.bind_tools(tools)

    def llm_node(state):
        """Main LLM processing node"""
        try:
            response = llm_with_tools.invoke(state['messages'])
            print(" TOOL CALL:", getattr(response, "tool_calls", None))
            return {'messages': state['messages'] + [response]}
        except Exception as e:
            error_msg = f"Error in LLM processing: {e}"
            logger.error(error_msg)
            return {'messages': state['messages'] + [AIMessage(content=error_msg)]}

    def router(state):
        """Route to appropriate next step based on the last message"""
        last_message = state['messages'][-1]

        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'tools'

        if hasattr(last_message, 'content'):
            content = last_message.content.lower()
            tool_keywords = [
                'system', 'metrics', 'monitor', 'docker', 'container',
                'security', 'audit', 'deploy', 'git', 'commit', 'test',
                'email', 'alert', 'status', 'report', 'traceback','summarize',
                'uid'
            ]
            if any(keyword in content for keyword in tool_keywords):
                return 'tools'

        return 'end'

    def tools_node(state):
        """Execute tools and return results"""
        try:
            tool_node = ToolNode(tools)
            result = tool_node.invoke(state)
            return {'messages': state['messages'] + result['messages']}
        except Exception as e:
            error_msg = f"Error executing tools: {e}"
            logger.error(error_msg)
            return {'messages': state['messages'] + [AIMessage(content=error_msg)]}

    # Build the state graph
    builder = StateGraph(ChatState)
    builder.add_node('llm', llm_node)
    builder.add_node('tools', tools_node)

    # Add edges
    builder.add_edge(START, 'llm')
    builder.add_edge('tools', 'llm')
    builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

    return builder.compile()

def start_background_monitoring():
    """Start background monitoring thread"""
    def monitoring_loop():
        while assistant.monitoring_active:
            try:
                # System monitoring
                assistant._check_system_health()

                # Docker monitoring if available
                if assistant.docker_client:
                    assistant._check_docker_health()

                # Git repository monitoring
                assistant._check_git_health()

                # Schedule any automated tasks
                schedule.run_pending()

                # Wait for next monitoring cycle
                time.sleep(assistant.config['monitoring_interval'])
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer if there's an error

    if not assistant.monitoring_active:
        assistant.monitoring_active = True
        assistant.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        assistant.monitoring_thread.start()
        logger.info("Background monitoring started")

def stop_background_monitoring():
    """Stop background monitoring"""
    assistant.monitoring_active = False
    if hasattr(assistant, 'monitoring_thread') and assistant.monitoring_thread:
        assistant.monitoring_thread.join(timeout=5)
    logger.info("Background monitoring stopped")

# Add methods to DevOpsAssistant class
def _check_system_health(self):
    """Check system health metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check thresholds and create alerts
        if cpu_percent > self.thresholds['cpu']:
            self.add_alert(Priority.HIGH, "System", f'CPU usage at {cpu_percent:.1f}%')

        if memory.percent > self.thresholds['memory']:
            self.add_alert(Priority.HIGH, "System", f"Memory usage at {memory.percent:.1f}%")

        if disk.percent > self.thresholds['disk']:
            self.add_alert(Priority.CRITICAL, "System", f'Disk usage at {disk.percent:.1f}%')

        # Check for high load average on Unix systems
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            if load_avg > cpu_count * 2:  # Load average is twice the CPU count
                self.add_alert(Priority.HIGH, "System", f"High load average: {load_avg:.2f}")
    except Exception as e:
        logger.error(f"Error checking system health: {e}")

def _check_docker_health(self):
    """Check Docker container health"""
    try:
        containers = self.docker_client.containers.list(all=True)
        for container in containers:
            # Check if container should be running but isn't
            if container.status != "running":
                # Only alert for containers that were recently running
                if container.attrs.get('State', {}).get('StartedAt'):
                    started_at = datetime.fromisoformat(
                        container.attrs['State']['StartedAt'].replace('Z', '+00:00')
                    )
                    if datetime.now(started_at.tzinfo) - started_at < timedelta(hours=1):
                        self.add_alert(
                            Priority.MEDIUM,
                            "Docker",
                            f"Container {container.name} is {container.status}"
                        )

            # Check container resource usage if running
            if container.status == "running":
                try:
                    stats = container.stats(stream=False)

                    # Calculate CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                   stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

                    # Calculate memory usage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0

                    # Alert on high resource usage
                    if cpu_percent > 90:
                        self.add_alert(
                            Priority.HIGH,
                            "Docker",
                            f"Container {container.name} high CPU usage: {cpu_percent:.1f}%"
                        )

                    if memory_percent > 90:
                        self.add_alert(
                            Priority.HIGH,
                            "Docker",
                            f"Container {container.name} high memory usage: {memory_percent:.1f}%"
                        )
                except Exception as e:
                    logger.debug(f"Could not get stats for container {container.name}: {e}")
    except Exception as e:
        logger.error(f"Error checking Docker health: {e}")

def _check_git_health(self):
    """Check Git repository health"""
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            return  # Not a git repository

        # Check for uncommitted changes that have been sitting too long
        status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                     capture_output=True, text=True)
        if status_result.stdout.strip():
            # Check last commit time
            last_commit = subprocess.run(['git', 'log', '-1', '--format=%ct'], 
                                       capture_output=True, text=True)
            if last_commit.returncode == 0:
                last_commit_time = datetime.fromtimestamp(int(last_commit.stdout.strip()))
                if datetime.now() - last_commit_time > timedelta(days=1):
                    self.add_alert(
                        Priority.LOW,
                        "Git",
                        "Uncommitted changes present for more than 24 hours"
                    )

        # Check if local branch is behind remote
        try:
            fetch_result = subprocess.run(['git', 'fetch', '--dry-run'],
                                        capture_output=True, text=True, timeout=10)
            if fetch_result.stderr:  # fetch --dry-run outputs to stderr
                self.add_alert(
                    Priority.LOW,
                    "Git",
                    "Local branch may be behind remote"
                )
        except subprocess.TimeoutExpired:
            pass  # Network might be slow, don't alert
    except Exception as e:
        logger.debug(f"Error checking Git health: {e}")

def start_monitoring(self):
    """Start background monitoring"""
    if not hasattr(self, 'monitoring_active'):
        self.monitoring_active = False
        
    if not self.monitoring_active:
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Background monitoring started")

def stop_monitoring(self):
    """Stop background monitoring"""
    self.monitoring_active = False
    if hasattr(self, 'monitoring_thread') and self.monitoring_thread:
        self.monitoring_thread.join(timeout=5)
    logger.info("Background monitoring stopped")

def _monitoring_loop(self):
    """Main monitoring loop"""
    while self.monitoring_active:
        try:
            # System monitoring
            self._check_system_health()

            # Docker monitoring if available
            if self.docker_client:
                self._check_docker_health()

            # Git repository monitoring
            self._check_git_health()

            # Schedule any automated tasks
            schedule.run_pending()

            # Wait for next monitoring cycle
            time.sleep(self.config['monitoring_interval'])
        except Exception as e:
            logger.error(f'Error in monitoring loop: {e}')
            time.sleep(60)  # Wait longer if there's an error

# Attach methods to class
DevOpsAssistant._check_system_health = _check_system_health
DevOpsAssistant._check_docker_health = _check_docker_health
DevOpsAssistant._check_git_health = _check_git_health
DevOpsAssistant.start_monitoring = start_monitoring
DevOpsAssistant.stop_monitoring = stop_monitoring
DevOpsAssistant._monitoring_loop = _monitoring_loop

# Reinitialize the assistant
assistant = DevOpsAssistant()

# Dashboard and UI functions
def display_dashboard():
    """Display a real-time dashboard"""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )

    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    def make_dashboard():
        # Header
        layout["header"].update(
            Panel(
                f"[bold green]DevOps Assistant Dashboard[/bold green] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="blue"
            )
        )

        # System metrics
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        metrics_table = Table(title="System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        metrics_table.add_column("Status", style="green")

        cpu_status = "🔴 HIGH" if cpu > 80 else "🟠 MEDIUM" if cpu > 60 else "🟢 OK"
        memory_status = "🔴 HIGH" if memory.percent > 85 else "🟠 MEDIUM" if memory.percent > 70 else "🟢 OK"
        disk_status = "🔴 HIGH" if disk.percent > 90 else "🟠 MEDIUM" if disk.percent > 80 else "🟢 OK"

        metrics_table.add_row("CPU", f"{cpu:.1f}%", cpu_status)
        metrics_table.add_row("Memory", f"{memory.percent:.1f}%", memory_status)
        metrics_table.add_row("Disk", f"{disk.percent:.1f}%", disk_status)
        layout["left"].update(metrics_table)

        # Recent alerts
        recent_alerts = assistant.alerts[-5:] if assistant.alerts else []
        alerts_table = Table(title="Recent Alerts")
        alerts_table.add_column("Time", style="cyan")
        alerts_table.add_column("Level", style="red")
        alerts_table.add_column("Source", style="yellow")
        alerts_table.add_column("Message", style="white")

        for alert in recent_alerts:
            level_color = {
                Priority.LOW: "green",
                Priority.MEDIUM: "yellow",
                Priority.HIGH: "red",
                Priority.CRITICAL: "bold red"
            }.get(alert.level, "white")
            
            alerts_table.add_row(
                alert.timestamp.strftime("%H:%M:%S"),
                f"[{level_color}]{alert.level.name}[/{level_color}]",
                alert.source,
                alert.message[:50] + "..." if len(alert.message) > 50 else alert.message
            )
        layout["right"].update(alerts_table)

        # Footer
        monitoring_status = "🟢 ACTIVE" if assistant.monitoring_active else "🔴 INACTIVE"
        layout["footer"].update(
            Panel(
                f"Monitoring: {monitoring_status} | Alerts: {len(assistant.alerts)} | Press Ctrl+C to exit",
                style="blue"
            )
        )
        return layout

    try:
        with Live(make_dashboard(), refresh_per_second=1, screen=True) as live:
            while True:
                live.update(make_dashboard())
                time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")

def interactive_chat():
    console.print(Panel("[bold green]DevOps Assistant Interactive Chat[/bold green]\nType 'quit' to exit, 'help' for commands"))

    direct = {
        'system metrics': get_system_metrics,
        'docker status': monitor_docker_containers,
        'git status': advanced_git_analysis,
        'run tests': run_tests,
        'security audit': security_audit,
        'deploy': automated_deployment,
        'report': generate_project_report,
        'email': list_unread_emails,
        'alerts': get_active_alerts,
    }

    while True:
        user = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        cmd = user.lower()

        if cmd in ('quit','exit','bye'):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if cmd == 'help':
            console.print(
                "[bold]Commands:[/bold]\n"
                "- system metrics\n- docker status\n- git status\n"
                "- run tests\n- security audit\n- deploy\n- report\n"
                "- email\n- summarize email <UID>\n- alerts\n"
                "- dashboard\n- start monitoring\n- stop monitoring\n- quit"
            )
            continue
        if cmd == 'dashboard':
            display_dashboard(); continue
        if cmd == 'start monitoring':
            assistant.start_monitoring(); console.print("[green]Monitoring started[/green]"); continue
        if cmd == 'stop monitoring':
            assistant.stop_monitoring(); console.print("[red]Monitoring stopped[/red]"); continue

        # Direct tool calls
        if cmd in direct:
            console.print(f"[bold blue]{cmd.title()}[/bold blue]")
            try:
                out = direct[cmd].invoke("")   # call LangChain tool
                console.print(out)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        # Summarize email special case
        if cmd.startswith('summarize email'):
            parts = cmd.split()
            if len(parts) == 3 and parts[2].isdigit():
                console.print("[bold blue]Summarizing Email[/bold blue]")
                console.print(summarize_email.invoke(parts[2]))
            else:
                console.print("[red]Usage: summarize email <UID>[/red]")
            continue

        # Fallback to raw LLM
        try:
            with console.status("[bold green]Thinking..."):
                resp = assistant.raw_llm.invoke([HumanMessage(content=user)])
                console.print(f"[bold green]Assistant:[/bold green] {resp.content}")
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")

def main():
    """Main entry point"""
    console.print(Panel("[bold blue]Advanced DevOps Assistant[/bold blue] - Starting up..."))

    # Setup scheduled tasks
    schedule.every(1).hours.do(lambda: assistant.add_alert(Priority.LOW, "Schedule", "Hourly health check"))
    schedule.every().day.at("09:00").do(lambda: console.print("Daily standup reminder!"))

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'monitor':
            console.print("[green]Starting monitoring mode...[/green]")
            assistant.start_monitoring()
            try:
                display_dashboard()
            except KeyboardInterrupt:
                pass
            finally:
                assistant.stop_monitoring()
        
        elif command == 'email' or 'unread email' in command:
                console.print("[bold blue]Checking Emails[/bold blue]")
                result = list_unread_emails.invoke({"limit": 3})
                console.print(result)
        elif command.startswith('summarize email'):
            uid = command.split()[-1] if len(command.split()) > 2 else ""
            if uid.isdigit():
                result = summarize_email.invoke({"uid": uid})
                console.print(result)
            else:
                console.print("[red]Usage: summarize email <UID>[/red]")
                
        elif command == 'chat':
            interactive_chat()
        elif command == 'report':
            console.print("[blue]Generating project report...[/blue]")
            report = generate_project_report()
            console.print(report)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Available commands: monitor, chat, report")
    else:
        interactive_chat()

if __name__ == "__main__":
    raw_llm = assistant.raw_llm
    llm = raw_llm.bind_tools([
    list_unread_emails,
    summarize_email
])
    main()


# import os
# import sys
# import json
# import asyncio
# import logging
# import subprocess
# import threading
# import time
# from datetime import datetime, timedelta
# from pathlib import Path
# from typing import Type, Dict, List, Optional, Any, TypedDict
# from dataclasses import dataclass
# from enum import Enum
# import re

# import psutil
# import docker
# import requests
# from rich.console import Console
# from rich.table import Table
# from rich.progress import Progress, SpinnerColumn, TextColumn
# from rich.panel import Panel
# from rich.layout import Layout
# from rich.live import Live
# from rich.prompt import Prompt
# from dotenv import load_dotenv
# from imap_tools import MailBox, AND
# import schedule
# import yaml
# from pydantic import BaseModel
# from langchain.chat_models import init_chat_model
# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.prebuilt import ToolNode
# from langgraph.graph import StateGraph, END, START
# from email_tools import list_unread_emails, summarize_email

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("devops_assistant.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# console = Console()

# class Priority(Enum):
#     LOW = 1
#     MEDIUM = 2
#     HIGH = 3
#     CRITICAL = 4

# @dataclass
# class Alert:
#     timestamp: datetime
#     level: Priority
#     source: str
#     message: str
#     resolved: bool = False

# @dataclass
# class SystemMetrics:
#     cpu_percent: float
#     memory_percent: float
#     disk_percent: float
#     network_io: Dict[str, int]
#     processes: int
#     uptime: float

# class ChatState(TypedDict):
#     messages: List[Dict[str, Any]]
#     context: Dict[str, Any]
#     alerts: List[Alert]
#     metrics: Optional[SystemMetrics]

# class DevOpsAssistant:
#     def __init__(self):
#         self.config = self._load_config()
#         self.alerts: List[Alert] = []
#         self.monitoring_active = False
#         self.docker_client = None

#         # Initialize LLM
#         self.raw_llm = init_chat_model(
#             self.config['chat_model'],
#             model_provider='ollama'
#         )

#         # Setup monitoring thresholds
#         self.thresholds = {
#             'cpu': self.config.get('cpu_threshold', 80),
#             'memory': self.config.get('memory_threshold', 85),
#             'disk': self.config.get('disk_threshold', 90)
#         }

#         # Initialize Docker client if available
#         try:
#             self.docker_client = docker.from_env()
#             logger.info("Docker client initialized successfully")
#         except Exception as e:
#             logger.warning(f"Docker not available: {e}")

#     def _load_config(self) -> Dict[str, Any]:
#         """Load configuration from environment and config file"""
#         config = {
#             'chat_model': os.getenv('CHAT_MODEL', 'qwen3:8b'),
#             'imap_host': os.getenv('IMAP_HOST'),
#             'imap_user': os.getenv('IMAP_USER'),
#             'imap_password': os.getenv('IMAP_PASSWORD'),
#             'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', '30')),
#             'alert_webhook': os.getenv('ALERT_WEBHOOK'),
#             'cpu_threshold': float(os.getenv('CPU_THRESHOLD', '80')),
#             'memory_threshold': float(os.getenv('MEMORY_THRESHOLD', '85')),
#             'disk_threshold': float(os.getenv('DISK_THRESHOLD', '90')),
#         }

#         # Load additional config from file if exists
#         config_file = Path('devops_config.yaml')
#         if config_file.exists():
#             with open(config_file, 'r') as f:
#                 file_config = yaml.safe_load(f)
#                 config.update(file_config)

#         return config

#     def add_alert(self, level: Priority, source: str, message: str):
#         """Add a new alert to the system"""
#         alert = Alert(
#             timestamp=datetime.now(),
#             level=level,
#             source=source,
#             message=message
#         )
#         self.alerts.append(alert)
#         logger.warning(f"Alert [{level.name}] from {source}: {message}")

#         # Send webhook notification if configured
#         if self.config.get('alert_webhook'):
#             self._send_webhook_alert(alert)

#     def _send_webhook_alert(self, alert: Alert):
#         """Send alert to configured webhook"""
#         try:
#             payload = {
#                 'timestamp': alert.timestamp.isoformat(),
#                 'level': alert.level.name,
#                 'source': alert.source,
#                 'message': alert.message
#             }
#             requests.post(self.config['alert_webhook'], json=payload, timeout=5)
#         except Exception as e:
#             logger.error(f"Failed to send webhook alert: {e}")

# # Initialize the assistant GLOBALLY so tools can access it
# assistant = DevOpsAssistant()

# # Email helper functions (from your working email_tools.py)
# def get_imap_config():
#     """Get IMAP configuration from assistant"""
#     return (
#         assistant.config.get("imap_host"),
#         assistant.config.get("imap_user"),
#         assistant.config.get("imap_password")
#     )

# # Enhanced Tools
# @tool
# def get_system_metrics() -> str:
#     """Get comprehensive system metrics including CPU, memory, disk, network, and processes."""
#     try:
#         # CPU metrics
#         cpu_percent = psutil.cpu_percent(interval=1)
#         cpu_count = psutil.cpu_count()
#         cpu_freq = psutil.cpu_freq()

#         # Memory metrics
#         memory = psutil.virtual_memory()
#         swap = psutil.swap_memory()

#         # Disk metrics
#         disk_usage = psutil.disk_usage('/')
#         disk_io = psutil.disk_io_counters()

#         # Network metrics
#         network_io = psutil.net_io_counters()

#         # Process metrics
#         processes = len(psutil.pids())
#         boot_time = psutil.boot_time()
#         uptime = time.time() - boot_time

#         # Check thresholds and create alerts
#         if cpu_percent > assistant.thresholds['cpu']:
#             assistant.add_alert(Priority.HIGH, "CPU", f"CPU usage at {cpu_percent}%")
#         if memory.percent > assistant.thresholds['memory']:
#             assistant.add_alert(Priority.HIGH, "Memory", f"Memory usage at {memory.percent}%")
#         if disk_usage.percent > assistant.thresholds['disk']:
#             assistant.add_alert(Priority.CRITICAL, "Disk", f"Disk usage at {disk_usage.percent}%")
            
#         metrics = {
#             "cpu": {
#                 "percent": cpu_percent,
#                 "count": cpu_count,
#                 "frequency": cpu_freq.current if cpu_freq else "N/A"
#             },
#             "memory": {
#                 "percent": memory.percent,
#                 "used_gb": round(memory.used / (1024**3), 2),
#                 "total_gb": round(memory.total / (1024**3), 2),
#                 "available_gb": round(memory.available / (1024**3), 2)
#             },
#             "swap": {
#                 "percent": swap.percent,
#                 "used_gb": round(swap.used / (1024**3), 2),
#                 "total_gb": round(swap.total / (1024**3), 2)
#             },
#             "disk": {
#                 "percent": disk_usage.percent,
#                 "used_gb": round(disk_usage.used / (1024**3), 2),
#                 "total_gb": round(disk_usage.total / (1024**3), 2),
#                 "free_gb": round(disk_usage.free / (1024**3), 2)
#             },
#             "network": {
#                 "bytes_sent": network_io.bytes_sent,
#                 "bytes_recv": network_io.bytes_recv,
#                 "packets_sent": network_io.packets_sent,
#                 "packets_recv": network_io.packets_recv
#             },
#             "system": {
#                 "processes": processes,
#                 "uptime_hours": round(uptime / 3600, 2)
#             }
#         }
#         return json.dumps(metrics, indent=2)
#     except Exception as e:
#         return f"Error getting system metrics: {e}"

# @tool
# def monitor_docker_containers():
#     """
#     Monitors active Docker containers and reports their CPU and memory usage.
#     Handles missing 'system_cpu_usage' from newer Docker Engine APIs.
#     """
#     try:
#         if not assistant.docker_client:
#             return "🐳 Docker client not available."
            
#         containers = assistant.docker_client.containers.list()
#         if not containers:
#             return "🐳 No running Docker containers found."

#         output = []
#         for container in containers:
#             stats = container.stats(stream=False)
#             name = container.name
#             status = container.status
#             mem_usage = stats['memory_stats']['usage']
#             mem_limit = stats['memory_stats'].get('limit', 1)
#             mem_percent = (mem_usage / mem_limit) * 100 if mem_limit else 0

#             cpu_percent = 0.0
#             try:
#                 cpu_stats = stats['cpu_stats']
#                 precpu_stats = stats['precpu_stats']

#                 # Newer Docker versions don't include 'system_cpu_usage'
#                 cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
#                 system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)

#                 if cpu_delta > 0 and system_delta > 0:
#                     cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage'].get('percpu_usage', [])) * 100
#             except Exception:
#                 cpu_percent = 0.0  # fail-safe

#             output.append(f"🟢 {name} | Status: {status} | CPU: {cpu_percent:.2f}% | Memory: {mem_percent:.2f}%")

#         return "\n".join(output)
#     except Exception as e:
#         return f"❌ Error monitoring Docker containers: {e}"


# @tool
# def advanced_git_analysis() -> str:
#     """Perform advanced Git repository analysis including branch status, commit trends, and code quality metrics."""
#     try:
#         analysis = {}

#         # Basic git status
#         status_result = subprocess.run(['git', 'status', '--porcelain'],
#                                        capture_output=True, text=True, check=True)
#         analysis['uncommitted_changes'] = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0

#         # Branch information
#         branch_result = subprocess.run(['git', 'branch', '-r'],
#                                        capture_output=True, text=True, check=True)
#         analysis['remote_branches'] = len([b for b in branch_result.stdout.split('\n') if b.strip()])

#         # Commit statistics (last 30 days)
#         since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
#         commits_result = subprocess.run(['git', 'log', '--since', since_date, '--oneline'],
#                                         capture_output=True, text=True, check=True)
#         analysis['commits_last_30_days'] = len(commits_result.stdout.strip().split('\n')) if commits_result.stdout.strip() else 0

#         # Contributors
#         contributors_result = subprocess.run(['git', 'shortlog', '-sn', '--since', since_date],
#                                             capture_output=True, text=True, check=True)
#         analysis['active_contributors'] = len(contributors_result.stdout.strip().split('\n')) if contributors_result.stdout.strip() else 0

#         # Repository size
#         size_result = subprocess.run(['git', 'count-objects', '-vH'],
#                                      capture_output=True, text=True, check=True)
#         analysis['repo_info'] = size_result.stdout

#         # Check for large files
#         large_files_result = subprocess.run(['find', '.', '-size', '+10M', '-type', 'f'],
#                                            capture_output=True, text=True)
#         large_files = [f for f in large_files_result.stdout.split('\n') if f.strip() and not f.startswith('./.git/')]
#         analysis['large_files_count'] = len(large_files)

#         if large_files:
#             assistant.add_alert(Priority.MEDIUM, "Git", f"Found {len(large_files)} large files in repository")

#         return json.dumps(analysis, indent=2)
#     except subprocess.CalledProcessError as e:
#         return f"Git analysis error: {e}"
#     except Exception as e:
#         return f"Error performing git analysis: {e}"

# @tool
# def security_audit() -> str:
#     """Perform a basic security audit of the system and codebase."""
#     audit_results = {}
#     try:
#         # Check for common security issues in Python files
#         security_issues = []

#         for root, dirs, files in os.walk('.'):
#             # Skip .git and other hidden directories
#             dirs[:] = [d for d in dirs if not d.startswith('.')]
            
#             for file in files:
#                 if file.endswith('.py'):
#                     file_path = os.path.join(root, file)
#                     try:
#                         with open(file_path, 'r', encoding='utf-8') as f:
#                             content = f.read()

#                         # Check for common security anti-patterns
#                         if 'eval(' in content:
#                             security_issues.append(f"{file_path}: Uses eval() - potential code injection")
#                         if 'exec(' in content:
#                             security_issues.append(f"{file_path}: Uses exec() - potential code injection")
#                         if 'shell=True' in content:
#                             security_issues.append(f"{file_path}: Uses shell=True - potential command injection")
#                         if 'password' in content.lower() and '=' in content:
#                             security_issues.append(f"{file_path}: Potential hardcoded password")
#                     except Exception:
#                         continue

#         audit_results['security_issues'] = security_issues

#         # Check file permissions
#         sensitive_files = ['.env', 'config.yaml', 'devops_config.yaml']
#         permission_issues = []

#         for file in sensitive_files:
#             if os.path.exists(file):
#                 stat_info = os.stat(file)
#                 permissions = oct(stat_info.st_mode)[-3:]
#                 if permissions != '600':
#                     permission_issues.append(f"{file}: Permissions {permissions} (should be 600)")

#         audit_results['permission_issues'] = permission_issues

#         # Create alerts for critical issues
#         critical_issues = len([issue for issue in security_issues if 'injection' in issue])
#         if critical_issues > 0:
#             assistant.add_alert(Priority.CRITICAL, "Security", f"Found {critical_issues} potential injection vulnerabilities")

#         return json.dumps(audit_results, indent=2)
#     except Exception as e:
#         return f"Error performing security audit: {e}"

# @tool
# def automated_deployment() -> str:
#     """Simulate automated deployment with pre-deployment checks."""
#     deployment_steps = []
#     try:
#         # Pre-deployment checks
#         deployment_steps.append("Starting pre-deployment checks...")

#         # Check git status
#         git_result = subprocess.run(['git', 'status', '--porcelain'],
#                                    capture_output=True, text=True)
#         if git_result.stdout.strip():
#             deployment_steps.append("❌ Uncommitted changes detected")
#             return '\n'.join(deployment_steps)
#         deployment_steps.append("✅ Git repository is clean")

#         # Check system resources
#         cpu_percent = psutil.cpu_percent(interval=1)
#         memory_percent = psutil.virtual_memory().percent

#         if cpu_percent > 80 or memory_percent > 80:
#             deployment_steps.append("⚠️ High system resource usage detected")
#             assistant.add_alert(Priority.MEDIUM, "Deployment", "High resource usage during deployment")
#         deployment_steps.append("✅ System resources check passed")

#         # Simulate deployment steps
#         deployment_steps.append("🚀 Starting deployment...")
#         deployment_steps.append("🔨 Building application...")
#         time.sleep(1)  # Simulate build time
#         deployment_steps.append("✅ Build completed successfully")

#         deployment_steps.append("📦 Deploying to staging...")
#         time.sleep(1)  # Simulate deployment time
#         deployment_steps.append("✅ Staging deployment completed")

#         deployment_steps.append("🧪 Running smoke tests...")
#         time.sleep(1)  # Simulate test time
#         deployment_steps.append("✅ Smoke tests passed")

#         deployment_steps.append("🎉 Deployment completed successfully")

#         return '\n'.join(deployment_steps)
#     except Exception as e:
#         deployment_steps.append(f"❌ Deployment failed: {e}")
#         return '\n'.join(deployment_steps)

# @tool
# def generate_project_report() -> str:
#     """Generate a comprehensive project health report."""
#     try:
#         report = {
#             "timestamp": datetime.now().isoformat(),
#             "project_name": os.path.basename(os.getcwd()),
#             "git_analysis": {},
#             "system_metrics": {},
#             "alerts_summary": {},
#             "recommendations": []
#         }

#         # Git analysis
#         try:
#             git_log = subprocess.run(['git', 'log', '--oneline', '-10'],
#                                     capture_output=True, text=True, check=True)
#             report['git_analysis']['recent_commits'] = len(git_log.stdout.strip().split('\n'))
#             git_status = subprocess.run(['git', 'status', '--porcelain'],
#                                       capture_output=True, text=True, check=True)
#             report['git_analysis']['uncommitted_files'] = len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
#         except:
#             report['git_analysis']['error'] = 'Git not available or not a git repository'

#         # System metrics
#         cpu = psutil.cpu_percent(interval=1)
#         memory = psutil.virtual_memory()
#         disk = psutil.disk_usage('/')

#         report["system_metrics"] = {
#             "cpu_percent": cpu,
#             "memory_percent": memory.percent,
#             "disk_percent": disk.percent,
#             "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2)
#         }

#         # Alerts summary
#         recent_alerts = [alert for alert in assistant.alerts
#                          if alert.timestamp > datetime.now() - timedelta(hours=24)]

#         report["alerts_summary"] = {
#             "total_alerts_24h": len(recent_alerts),
#             "critical_alerts": len([a for a in recent_alerts if a.level == Priority.CRITICAL]),
#             "high_alerts": len([a for a in recent_alerts if a.level == Priority.HIGH]),
#             "unresolved_alerts": len([a for a in recent_alerts if not a.resolved])
#         }
        
#         if cpu > 80:
#             report["recommendations"].append("Consider optimizing CPU-intensive processes")
#         if memory.percent > 85:
#             report["recommendations"].append("Memory usage is high - consider adding more RAM or optimizing memory usage")
#         if disk.percent > 90:
#             report["recommendations"].append("Disk space is critically low - cleanup required")
#         if report["git_analysis"].get("uncommitted_files", 0) > 0:
#             report["recommendations"].append("Commit or stash uncommitted changes")

#         return json.dumps(report, indent=2)
#     except Exception as e:
#         return f"Error generating project report: {e}"

# @tool
# def check_git_status() -> str:
#     """Run 'git status' and return the result with additional analysis."""
#     try:
#         result = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True)
#         status_output = result.stdout

#         # Additional git info
#         branch_result = subprocess.run(['git', 'branch', '--show-current'],
#                                       capture_output=True, text=True, check=True)
#         current_branch = branch_result.stdout.strip()

#         try:
#             ahead_behind = subprocess.run(['git', 'status', '-b', '--porcelain'],
#                                          capture_output=True, text=True, check=True)
#             branch_info = ahead_behind.stdout.split('\n')[0] if ahead_behind.stdout else ""
#         except:
#             branch_info = ""
#         enhanced_status = f"Current branch: {current_branch}\n"
#         if branch_info:
#             enhanced_status += f"Branch Status: {branch_info}\n"
#         enhanced_status += f"\n{status_output}"
#         return enhanced_status
#     except subprocess.CalledProcessError as e:
#         return f"Error running git status: {e}"

# @tool
# def list_recent_commits(n: int = 5) -> str:
#     """List the most recent n git commits with enhanced information."""
#     try:
#         cmd = ['git', 'log', f'--max-count={n}', '--pretty=format:%h - %s (%an, %ar) [%d]', '--graph']
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         return f'Error fetching commits: {e}'

# @tool
# def run_tests() -> str:
#     """Fixed test runner that excludes venv and only runs actual project tests."""
#     try:
#         # Find only project test files (exclude venv, site-packages)
#         test_files = []
#         exclude_dirs = {'venv', 'site-packages', '__pycache__', '.git', 'node_modules'}
        
#         for root, dirs, files in os.walk('.'):
#             # Remove excluded directories
#             dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
#             for file in files:
#                 if ((file.startswith('test_') or file.endswith('_test.py')) and 
#                     file.endswith('.py') and 
#                     not any(ex in root for ex in exclude_dirs)):
#                     test_files.append(os.path.join(root, file))
        
#         if not test_files:
#             return "❌ No project test files found. Create test_*.py files in your project directory."
        
#         # Validate test files have actual test functions
#         valid_test_files = []
#         for test_file in test_files:
#             try:
#                 with open(test_file, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     # Check if file contains test functions
#                     if 'def test_' in content or 'class Test' in content:
#                         valid_test_files.append(test_file)
#                     else:
#                         logger.warning(f"Test file {test_file} contains no test functions")
#             except Exception as e:
#                 logger.warning(f"Could not read test file {test_file}: {e}")
        
#         if not valid_test_files:
#             return "❌ No valid test files found. Test files must contain functions starting with 'test_' or test classes."
        
#         # Run pytest with verbose output and discovery
#         cmd = [sys.executable, '-m', 'pytest', '-v', '--tb=short'] + valid_test_files[:5]
#         result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
#         output = f"🔍 Found {len(test_files)} test files, {len(valid_test_files)} valid\n"
#         output += f"🧪 Running tests on: {', '.join([Path(f).name for f in valid_test_files[:5]])}\n\n"
        
#         if result.returncode == 0:
#             output += "✅ All tests passed!\n"
#             output += result.stdout[-300:] if result.stdout else ""
#         else:
#             output += "❌ Tests failed:\n"
#             # Show both stdout and stderr for better debugging
#             if result.stdout:
#                 output += "STDOUT:\n" + result.stdout[-400:] + "\n"
#             if result.stderr:
#                 output += "STDERR:\n" + result.stderr[-400:] + "\n"
                
#         return output
        
#     except subprocess.TimeoutExpired:
#         return "❌ Tests timed out after 60 seconds"
#     except Exception as e:
#         return f"❌ Test execution error: {e}"

# @tool
# def explain_traceback(log: str) -> str:
#     """Provide a detailed explanation of a Python traceback log with suggested fixes."""
#     prompt = f"""
# Analyze this Python traceback and provide:
# 1. A clear explanation of what went wrong
# 2. The root cause of the error
# 3. Specific steps to fix the issue
# 4. Best practices to prevent similar errors

# Traceback:
# {log}
# """
#     response = assistant.raw_llm.invoke([HumanMessage(content=prompt)])
#     return response.content

# @tool
# def get_active_alerts() -> str:
#     """Get all active (unresolved) alerts from the system."""
#     active_alerts = [alert for alert in assistant.alerts if not alert.resolved]

#     if not active_alerts:
#         return "✅ No active alerts."

#     alert_data = []
#     for alert in active_alerts:
#         alert_data.append({
#             "timestamp": alert.timestamp.isoformat(),
#             "level": alert.level.name,
#             "source": alert.source,
#             "message": alert.message
#         })
#     return json.dumps(alert_data, indent=2)

# # Create tool list
# tools = [
#     get_system_metrics,
#     monitor_docker_containers,
#     advanced_git_analysis,
#     security_audit,
#     automated_deployment,
#     generate_project_report,
#     list_unread_emails,
#     summarize_email,
#     check_git_status,
#     list_recent_commits,
#     run_tests,
#     explain_traceback,
#     get_active_alerts
# ]

# # Agent Graph Setup
# def create_agent_graph():
#     """Create and configure the agent graph"""
#     # Bind tools to LLM
#     llm_with_tools = assistant.raw_llm.bind_tools(tools)

#     def llm_node(state):
#         """Main LLM processing node"""
#         try:
#             response = llm_with_tools.invoke(state['messages'])
#             return {'messages': state['messages'] + [response]}
#         except Exception as e:
#             error_msg = f"Error in LLM processing: {e}"
#             logger.error(error_msg)
#             return {'messages': state['messages'] + [AIMessage(content=error_msg)]}

#     def router(state):
#         """Route to appropriate next step based on the last message"""
#         last_message = state['messages'][-1]
        
#         if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#             return 'tools'
#         return 'end'

#     def tools_node(state):
#         """Execute tools and return results"""
#         try:
#             tool_node = ToolNode(tools)
#             result = tool_node.invoke(state)
#             return {'messages': state['messages'] + result['messages']}
#         except Exception as e:
#             error_msg = f"Error executing tools: {e}"
#             logger.error(error_msg)
#             return {'messages': state['messages'] + [AIMessage(content=error_msg)]}

   
#     builder = StateGraph(ChatState)
#     builder.add_node('llm', llm_node)
#     builder.add_node('tools', tools_node)

#     # Add edges
#     builder.add_edge(START, 'llm')
#     builder.add_edge('tools', 'llm')
#     builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

#     return builder.compile()

# # Initialize the agent graph globally
# agent_graph = create_agent_graph()

# def start_background_monitoring():
#     """Start background monitoring thread"""
#     def monitoring_loop():
#         while assistant.monitoring_active:
#             try:
#                 # System monitoring
#                 assistant._check_system_health()

#                 # Docker monitoring if available
#                 if assistant.docker_client:
#                     assistant._check_docker_health()

#                 # Git repository monitoring
#                 assistant._check_git_health()

#                 # Schedule any automated tasks
#                 schedule.run_pending()

#                 # Wait for next monitoring cycle
#                 time.sleep(assistant.config['monitoring_interval'])
#             except Exception as e:
#                 logger.error(f"Error in monitoring loop: {e}")
#                 time.sleep(60)  # Wait longer if there's an error

#     if not assistant.monitoring_active:
#         assistant.monitoring_active = True
#         assistant.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
#         assistant.monitoring_thread.start()
#         logger.info("Background monitoring started")

# def stop_background_monitoring():
#     """Stop background monitoring"""
#     assistant.monitoring_active = False
#     if hasattr(assistant, 'monitoring_thread') and assistant.monitoring_thread:
#         assistant.monitoring_thread.join(timeout=5)
#     logger.info("Background monitoring stopped")

# # Add methods to DevOpsAssistant class
# def _check_system_health(self):
#     """Check system health metrics"""
#     try:
#         cpu_percent = psutil.cpu_percent(interval=1)
#         memory = psutil.virtual_memory()
#         disk = psutil.disk_usage('/')

#         # Check thresholds and create alerts
#         if cpu_percent > self.thresholds['cpu']:
#             self.add_alert(Priority.HIGH, "System", f'CPU usage at {cpu_percent:.1f}%')

#         if memory.percent > self.thresholds['memory']:
#             self.add_alert(Priority.HIGH, "System", f"Memory usage at {memory.percent:.1f}%")

#         if disk.percent > self.thresholds['disk']:
#             self.add_alert(Priority.CRITICAL, "System", f'Disk usage at {disk.percent:.1f}%')

#         # Check for high load average on Unix systems
#         if hasattr(os, 'getloadavg'):
#             load_avg = os.getloadavg()[0]  # 1-minute load average
#             cpu_count = psutil.cpu_count()
#             if load_avg > cpu_count * 2:  # Load average is twice the CPU count
#                 self.add_alert(Priority.HIGH, "System", f"High load average: {load_avg:.2f}")
#     except Exception as e:
#         logger.error(f"Error checking system health: {e}")

# def _check_docker_health(self):
#     """Check Docker container health"""
#     try:
#         containers = self.docker_client.containers.list(all=True)
#         for container in containers:
#             # Check if container should be running but isn't
#             if container.status != "running":
#                 # Only alert for containers that were recently running
#                 if container.attrs.get('State', {}).get('StartedAt'):
#                     try:
#                         started_at = datetime.fromisoformat(
#                             container.attrs['State']['StartedAt'].replace('Z', '+00:00')
#                         )
#                         if datetime.now(started_at.tzinfo) - started_at < timedelta(hours=1):
#                             self.add_alert(
#                                 Priority.MEDIUM,
#                                 "Docker",
#                                 f"Container {container.name} is {container.status}"
#                             )
#                     except:
#                         pass  # Skip datetime parsing errors

#             # Check container resource usage if running
#             if container.status == "running":
#                 try:
#                     stats = container.stats(stream=False)

#                     # Calculate CPU usage (handle missing system_cpu_usage)
#                     cpu_percent = 0.0
#                     try:
#                         cpu_stats = stats['cpu_stats']
#                         precpu_stats = stats['precpu_stats']
#                         cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
#                         system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
                        
#                         if cpu_delta > 0 and system_delta > 0:
#                             cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage'].get('percpu_usage', [])) * 100
#                     except:
#                         cpu_percent = 0.0

#                     # Calculate memory usage
#                     memory_usage = stats['memory_stats'].get('usage', 0)
#                     memory_limit = stats['memory_stats'].get('limit', 1)
#                     memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0

#                     # Alert on high resource usage
#                     if cpu_percent > 90:
#                         self.add_alert(
#                             Priority.HIGH,
#                             "Docker",
#                             f"Container {container.name} high CPU usage: {cpu_percent:.1f}%"
#                         )

#                     if memory_percent > 90:
#                         self.add_alert(
#                             Priority.HIGH,
#                             "Docker",
#                             f"Container {container.name} high memory usage: {memory_percent:.1f}%"
#                         )
#                 except Exception as e:
#                     logger.debug(f"Could not get stats for container {container.name}: {e}")
#     except Exception as e:
#         logger.error(f"Error checking Docker health: {e}")

# def _check_git_health(self):
#     """Check Git repository health"""
#     try:
#         # Check if we're in a git repository
#         result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
#                                capture_output=True, text=True)
#         if result.returncode != 0:
#             return  # Not a git repository

#         # Check for uncommitted changes that have been sitting too long
#         status_result = subprocess.run(['git', 'status', '--porcelain'], 
#                                      capture_output=True, text=True)
#         if status_result.stdout.strip():
#             # Check last commit time
#             last_commit = subprocess.run(['git', 'log', '-1', '--format=%ct'], 
#                                        capture_output=True, text=True)
#             if last_commit.returncode == 0:
#                 last_commit_time = datetime.fromtimestamp(int(last_commit.stdout.strip()))
#                 if datetime.now() - last_commit_time > timedelta(days=1):
#                     self.add_alert(
#                         Priority.LOW,
#                         "Git",
#                         "Uncommitted changes present for more than 24 hours"
#                     )

#         # Check if local branch is behind remote
#         try:
#             fetch_result = subprocess.run(['git', 'fetch', '--dry-run'],
#                                         capture_output=True, text=True, timeout=10)
#             if fetch_result.stderr:  # fetch --dry-run outputs to stderr
#                 self.add_alert(
#                     Priority.LOW,
#                     "Git",
#                     "Local branch may be behind remote"
#                 )
#         except subprocess.TimeoutExpired:
#             pass  # Network might be slow, don't alert
#     except Exception as e:
#         logger.debug(f"Error checking Git health: {e}")

# def start_monitoring(self):
#     """Start background monitoring"""
#     if not hasattr(self, 'monitoring_active'):
#         self.monitoring_active = False
        
#     if not self.monitoring_active:
#         self.monitoring_active = True
#         self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
#         self.monitoring_thread.start()
#         logger.info("Background monitoring started")

# def stop_monitoring(self):
#     """Stop background monitoring"""
#     self.monitoring_active = False
#     if hasattr(self, 'monitoring_thread') and self.monitoring_thread:
#         self.monitoring_thread.join(timeout=5)
#     logger.info("Background monitoring stopped")

# def _monitoring_loop(self):
#     """Main monitoring loop"""
#     while self.monitoring_active:
#         try:
#             # System monitoring
#             self._check_system_health()

#             # Docker monitoring if available
#             if self.docker_client:
#                 self._check_docker_health()

#             # Git repository monitoring
#             self._check_git_health()

#             # Schedule any automated tasks
#             schedule.run_pending()

#             # Wait for next monitoring cycle
#             time.sleep(self.config['monitoring_interval'])
#         except Exception as e:
#             logger.error(f'Error in monitoring loop: {e}')
#             time.sleep(60)  # Wait longer if there's an error

# # Attach methods to class
# DevOpsAssistant._check_system_health = _check_system_health
# DevOpsAssistant._check_docker_health = _check_docker_health
# DevOpsAssistant._check_git_health = _check_git_health
# DevOpsAssistant.start_monitoring = start_monitoring
# DevOpsAssistant.stop_monitoring = stop_monitoring
# DevOpsAssistant._monitoring_loop = _monitoring_loop

# # Reinitialize the assistant
# assistant = DevOpsAssistant()

# # Dashboard and UI functions
# def display_dashboard():
#     """Display a real-time dashboard"""
#     layout = Layout()

#     layout.split_column(
#         Layout(name="header", size=3),
#         Layout(name="main", ratio=1),
#         Layout(name="footer", size=3)
#     )

#     layout["main"].split_row(
#         Layout(name="left"),
#         Layout(name="right")
#     )

#     def make_dashboard():
#         # Header
#         layout["header"].update(
#             Panel(
#                 f"[bold green]DevOps Assistant Dashboard[/bold green] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
#                 style="blue"
#             )
#         )

#         # System metrics
#         cpu = psutil.cpu_percent(interval=0.1)
#         memory = psutil.virtual_memory()
#         disk = psutil.disk_usage('/')
#         metrics_table = Table(title="System Metrics")
#         metrics_table.add_column("Metric", style="cyan")
#         metrics_table.add_column("Value", style="magenta")
#         metrics_table.add_column("Status", style="green")

#         cpu_status = "🔴 HIGH" if cpu > 80 else "🟠 MEDIUM" if cpu > 60 else "🟢 OK"
#         memory_status = "🔴 HIGH" if memory.percent > 85 else "🟠 MEDIUM" if memory.percent > 70 else "🟢 OK"
#         disk_status = "🔴 HIGH" if disk.percent > 90 else "🟠 MEDIUM" if disk.percent > 80 else "🟢 OK"

#         metrics_table.add_row("CPU", f"{cpu:.1f}%", cpu_status)
#         metrics_table.add_row("Memory", f"{memory.percent:.1f}%", memory_status)
#         metrics_table.add_row("Disk", f"{disk.percent:.1f}%", disk_status)
#         layout["left"].update(metrics_table)

#         # Recent alerts
#         recent_alerts = assistant.alerts[-5:] if assistant.alerts else []
#         alerts_table = Table(title="Recent Alerts")
#         alerts_table.add_column("Time", style="cyan")
#         alerts_table.add_column("Level", style="red")
#         alerts_table.add_column("Source", style="yellow")
#         alerts_table.add_column("Message", style="white")

#         for alert in recent_alerts:
#             level_color = {
#                 Priority.LOW: "green",
#                 Priority.MEDIUM: "yellow",
#                 Priority.HIGH: "red",
#                 Priority.CRITICAL: "bold red"
#             }.get(alert.level, "white")
            
#             alerts_table.add_row(
#                 alert.timestamp.strftime("%H:%M:%S"),
#                 f"[{level_color}]{alert.level.name}[/{level_color}]",
#                 alert.source,
#                 alert.message[:50] + "..." if len(alert.message) > 50 else alert.message
#             )
#         layout["right"].update(alerts_table)

#         # Footer
#         monitoring_status = "🟢 ACTIVE" if assistant.monitoring_active else "🔴 INACTIVE"
#         layout["footer"].update(
#             Panel(
#                 f"Monitoring: {monitoring_status} | Alerts: {len(assistant.alerts)} | Press Ctrl+C to exit",
#                 style="blue"
#             )
#         )
#         return layout

#     try:
#         with Live(make_dashboard(), refresh_per_second=1, screen=True) as live:
#             while True:
#                 live.update(make_dashboard())
#                 time.sleep(1)
#     except KeyboardInterrupt:
#         console.print("\n[yellow]Dashboard stopped by user[/yellow]")

# def interactive_chat():
#     """Enhanced interactive chat with agent graph integration"""
#     console.print(Panel("[bold green]DevOps Assistant Interactive Chat[/bold green]\nType 'quit' to exit, 'help' for commands"))

#     # Direct tool mappings for quick access - call functions directly
#     direct_tools = {
#         'system metrics': lambda: get_system_metrics(),
#         'docker status': lambda: monitor_docker_containers(),
#         'git status': lambda: advanced_git_analysis(),
#         'run tests': lambda: run_tests(),
#         'security audit': lambda: security_audit(),
#         'deploy': lambda: automated_deployment(),
#         'report': lambda: generate_project_report(),
#         'email': lambda: list_unread_emails(5),
#         'alerts': lambda: get_active_alerts(),
#     }

#     while True:
#         user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
#         cmd = user_input.lower()

#         if cmd in ('quit', 'exit', 'bye'):
#             console.print("[yellow]Goodbye![/yellow]")
#             break
            
#         if cmd == 'help':
#             console.print(
#                 "[bold]Commands:[/bold]\n"
#                 "- system metrics\n- docker status\n- git status\n"
#                 "- run tests\n- security audit\n- deploy\n- report\n"
#                 "- email\n- summarize email <UID>\n- alerts\n"
#                 "- dashboard\n- start monitoring\n- stop monitoring\n- quit\n\n"
#                 "[bold]Or ask any question - I'll use the appropriate tools![/bold]"
#             )
#             continue
            
#         if cmd == 'dashboard':
#             display_dashboard()
#             continue
            
#         if cmd == 'start monitoring':
#             assistant.start_monitoring()
#             console.print("[green]Monitoring started[/green]")
#             continue
            
#         if cmd == 'stop monitoring':
#             assistant.stop_monitoring()
#             console.print("[red]Monitoring stopped[/red]")
#             continue

#         # Direct tool calls for quick access - Fix: call functions directly
#         if cmd in direct_tools:
#             console.print(f"[bold blue]{cmd.title()}[/bold blue]")
#             try:
#                 result = direct_tools[cmd]()  # Call the lambda function
#                 console.print(result)
#             except Exception as e:
#                 console.print(f"[red]Error: {e}[/red]")
#             continue

#         # Special case for summarize email
#         if cmd.startswith('summarize email'):
#             parts = cmd.split()
#             if len(parts) == 3 and parts[2].isdigit():
#                 console.print("[bold blue]Summarizing Email[/bold blue]")
#                 result = summarize_email(parts[2])  # Call directly
#                 console.print(result)
#             else:
#                 console.print("[red]Usage: summarize email <UID>[/red]")
#             continue

#         # Use the agent graph for complex queries
#         try:
#             with console.status("[bold green]Processing with AI agent..."):
#                 # Create initial state
#                 initial_state = {
#                     'messages': [HumanMessage(content=user_input)],
#                     'context': {},
#                     'alerts': assistant.alerts,
#                     'metrics': None
#                 }
                
#                 # Run the agent
#                 result = agent_graph.invoke(initial_state)
                
#                 # Get the last AI message
#                 last_message = result['messages'][-1]
#                 if hasattr(last_message, 'content'):
#                     console.print(f"[bold green]Assistant:[/bold green] {last_message.content}")
#                 else:
#                     console.print(f"[bold green]Assistant:[/bold green] {str(last_message)}")
                    
#         except Exception as e:
#             # Fallback to raw LLM if agent fails
#             try:
#                 with console.status("[bold yellow]Fallback to basic AI..."):
#                     response = assistant.raw_llm.invoke([HumanMessage(content=user_input)])
#                     console.print(f"[bold green]Assistant:[/bold green] {response.content}")
#             except Exception as e2:
#                 console.print(f"[red]Error: {e2}[/red]")


# def main():
#     """Main entry point"""
#     console.print(Panel("[bold blue]Advanced DevOps Assistant[/bold blue] - Starting up..."))

#     try:
#         import schedule
#         schedule.every(1).hours.do(lambda: assistant.add_alert(Priority.LOW, "Schedule", "Hourly health check"))
#         schedule.every().day.at("09:00").do(lambda: console.print("Daily standup reminder!"))
#     except ImportError:
#         logger.warning("Schedule module not available - skipping scheduled tasks")

#     if len(sys.argv) > 1:
#         command = sys.argv[1].lower()

#         if command == 'monitor':
#             console.print("[green]Starting monitoring mode...[/green]")
#             assistant.start_monitoring()
#             try:
#                 display_dashboard()
#             except KeyboardInterrupt:
#                 pass
#             finally:
#                 assistant.stop_monitoring()

#         elif command in ['email', 'emails']:
#             console.print("[bold blue]Checking Emails[/bold blue]")
#             result = list_unread_emails.invoke({"limit": 5})
#             console.print(result)

#         elif command.startswith('summarize'):
#             if len(sys.argv) > 2 and sys.argv[2].isdigit():
#                 try:
#                     result = summarize_email.invoke(sys.argv[2])
#                 except Exception as e:
#                     result = f"❌ Error summarizing: {e}"
#                 console.print(result)
#             else:
#                 console.print("[red]Usage: python devops.py summarize <UID>[/red]")

#         elif command == 'chat':
#             interactive_chat()

#         elif command == 'report':
#             console.print("[blue]Generating project report...[/blue]")
#             report = generate_project_report()
#             console.print(report)

#         elif command == 'test':
#             console.print("[blue]Running tests...[/blue]")
#             result = run_tests()
#             console.print(result)

#         else:
#             console.print(f"[red]Unknown command: {command}[/red]")
#             console.print("Available commands: monitor, chat, report, test, email, summarize <UID>")
#     else:
#         interactive_chat()

# if __name__ == "__main__":
#     # Initialize the agent graph
#     try:
#         agent_graph = create_agent_graph()
#         logger.info("Agent graph initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize agent graph: {e}")
#         agent_graph = None
    
#     main()

