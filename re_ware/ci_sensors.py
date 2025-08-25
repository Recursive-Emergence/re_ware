"""
CI/Test Sensors for RE_ware
===========================

Enhanced sensors for CI/CD systems, test results, and coverage reports.
Parses GitHub Actions, JUnit XML, pytest results, and coverage data.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess
import asyncio

from .sensor_hub import SensorInterface, DomainEvent
from .ontology import NodeType, RelationType, create_node, create_edge


class GitHubActionsSensor(SensorInterface):
    """Monitor GitHub Actions workflows and runs"""
    
    def __init__(self, project_root: Path):
        super().__init__(name="github_actions", description="GitHub Actions CI/CD monitoring")
        self.project_root = project_root
        self.gh_available = self._check_gh_cli()
    
    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available"""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def sense(self) -> List[Dict[str, Any]]:
        """Sense GitHub Actions workflow runs"""
        if not self.gh_available:
            return []
        
        events = []
        
        try:
            # Get recent workflow runs
            result = await asyncio.create_subprocess_exec(
                "gh", "run", "list", "--limit", "20", "--json", 
                "conclusion,createdAt,event,headBranch,headSha,id,name,status,updatedAt,url,workflowId",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return events
            
            runs = json.loads(stdout.decode())
            
            for run in runs:
                # Create BUILD node for each workflow run
                run_id = f"build:{run['id']}"
                
                # Only process recent runs (avoid duplicates on subsequent polls)
                updated_at = datetime.fromisoformat(run['updatedAt'].replace('Z', '+00:00'))
                if updated_at.timestamp() <= self.last_scan_time:
                    continue
                
                events.append({
                    "event_type": "node_update",
                    "node": create_node(
                        NodeType.BUILD,
                        f"{run['name']} #{run['id']}",
                        node_id=run_id,
                        content={
                            "workflow_name": run['name'],
                            "run_id": run['id'],
                            "status": run['status'],
                            "conclusion": run.get('conclusion'),
                            "branch": run['headBranch'],
                            "commit_sha": run['headSha'],
                            "created_at": run['createdAt'],
                            "updated_at": run['updatedAt'],
                            "url": run['url'],
                            "event_trigger": run['event'],
                            "build_system": "github_actions"
                        }
                    )
                })
                
                # Create edge to project
                events.append({
                    "event_type": "edge_create",
                    "edge": create_edge(
                        RelationType.BELONGS_TO,
                        run_id,
                        "project:root"
                    )
                })
            
            self.update_watermark()
            
        except Exception as e:
            print(f"⚠️  GitHub Actions sensing error: {e}")
        
        return events


class JUnitTestSensor(SensorInterface):
    """Monitor JUnit XML test results"""
    
    def __init__(self, project_root: Path, patterns: List[str] = None):
        super().__init__(name="junit", description="JUnit XML test result monitoring")
        self.project_root = project_root
        self.patterns = patterns or [
            "build/test-results/**/*.xml",
            "target/surefire-reports/*.xml",
            "test-results/*.xml",
            "tests/results/*.xml"
        ]
    
    async def sense(self) -> List[Dict[str, Any]]:
        """Sense JUnit XML test results"""
        events = []
        
        for pattern in self.patterns:
            for xml_file in self.project_root.glob(pattern):
                if not xml_file.is_file():
                    continue
                
                # Check if file is newer than last scan
                if xml_file.stat().st_mtime <= self.last_scan_time:
                    continue
                
                try:
                    events.extend(self._parse_junit_xml(xml_file))
                except Exception as e:
                    print(f"⚠️  Failed to parse JUnit XML {xml_file}: {e}")
        
        if events:
            self.update_watermark()
        
        return events
    
    def _parse_junit_xml(self, xml_file: Path) -> List[Dict[str, Any]]:
        """Parse JUnit XML file and create test nodes/edges"""
        events = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Handle both <testsuite> and <testsuites> roots
            testsuites = []
            if root.tag == "testsuites":
                testsuites = root.findall("testsuite")
            elif root.tag == "testsuite":
                testsuites = [root]
            
            for testsuite in testsuites:
                suite_name = testsuite.get("name", "Unknown Suite")
                suite_id = f"testsuite:{abs(hash(suite_name + str(xml_file)))}"
                
                # Create test suite node
                events.append({
                    "event_type": "node_update",
                    "node": create_node(
                        NodeType.TESTSUITE,
                        f"Test Suite: {suite_name}",
                        node_id=suite_id,
                        content={
                            "suite_name": suite_name,
                            "tests": int(testsuite.get("tests", 0)),
                            "failures": int(testsuite.get("failures", 0)),
                            "errors": int(testsuite.get("errors", 0)),
                            "skipped": int(testsuite.get("skipped", 0)),
                            "time": float(testsuite.get("time", 0.0)),
                            "timestamp": testsuite.get("timestamp"),
                            "source_file": str(xml_file),
                            "test_framework": "junit"
                        }
                    )
                })
                
                # Link to project
                events.append({
                    "event_type": "edge_create",
                    "edge": create_edge(
                        RelationType.BELONGS_TO,
                        suite_id,
                        "project:root"
                    )
                })
                
                # Process individual test cases
                for testcase in testsuite.findall("testcase"):
                    test_name = testcase.get("name", "Unknown Test")
                    class_name = testcase.get("classname", "")
                    test_id = f"test:{abs(hash(class_name + test_name + str(xml_file)))}"
                    
                    # Determine test status
                    status = "passed"
                    failure_message = None
                    if testcase.find("failure") is not None:
                        status = "failed"
                        failure = testcase.find("failure")
                        failure_message = failure.get("message", "")
                    elif testcase.find("error") is not None:
                        status = "error"
                        error = testcase.find("error")
                        failure_message = error.get("message", "")
                    elif testcase.find("skipped") is not None:
                        status = "skipped"
                    
                    # Create test case node
                    test_content = {
                        "test_name": test_name,
                        "class_name": class_name,
                        "status": status,
                        "time": float(testcase.get("time", 0.0)),
                        "test_framework": "junit"
                    }
                    
                    if failure_message:
                        test_content["failure_message"] = failure_message
                    
                    events.append({
                        "event_type": "node_update",
                        "node": create_node(
                            NodeType.TEST,
                            f"{class_name}.{test_name}" if class_name else test_name,
                            node_id=test_id,
                            content=test_content
                        )
                    })
                    
                    # Link test to suite
                    events.append({
                        "event_type": "edge_create",
                        "edge": create_edge(
                            RelationType.BELONGS_TO,
                            test_id,
                            suite_id
                        )
                    })
        
        except ET.ParseError as e:
            print(f"⚠️  Invalid XML in {xml_file}: {e}")
        
        return events


class PytestCoverageSensor(SensorInterface):
    """Monitor pytest coverage reports"""
    
    def __init__(self, project_root: Path, patterns: List[str] = None):
        super().__init__(name="coverage", description="Python test coverage monitoring")
        self.project_root = project_root
        self.patterns = patterns or [
            "coverage.xml",
            "htmlcov/coverage.xml",
            ".coverage.xml",
            "reports/coverage.xml"
        ]
    
    async def sense(self) -> List[Dict[str, Any]]:
        """Sense coverage reports"""
        events = []
        
        for pattern in self.patterns:
            for coverage_file in self.project_root.glob(pattern):
                if not coverage_file.is_file():
                    continue
                
                # Check if file is newer than last scan
                if coverage_file.stat().st_mtime <= self.last_scan_time:
                    continue
                
                try:
                    events.extend(self._parse_coverage_xml(coverage_file))
                except Exception as e:
                    print(f"⚠️  Failed to parse coverage XML {coverage_file}: {e}")
        
        if events:
            self.update_watermark()
        
        return events
    
    def _parse_coverage_xml(self, xml_file: Path) -> List[Dict[str, Any]]:
        """Parse coverage XML and create coverage nodes"""
        events = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Overall coverage node
            coverage_id = f"coverage:{abs(hash(str(xml_file)))}"
            
            # Extract overall metrics
            overall_line_rate = float(root.get("line-rate", 0.0))
            overall_branch_rate = float(root.get("branch-rate", 0.0))
            
            events.append({
                "event_type": "node_update",
                "node": create_node(
                    NodeType.COVERAGE,
                    f"Coverage Report",
                    node_id=coverage_id,
                    content={
                        "type": "coverage_report",
                        "line_coverage": overall_line_rate * 100,
                        "branch_coverage": overall_branch_rate * 100,
                        "timestamp": datetime.now().isoformat(),
                        "source_file": str(xml_file),
                        "tool": "coverage.py"
                    }
                )
            })
            
            # Link to project
            events.append({
                "event_type": "edge_create",
                "edge": create_edge(
                    RelationType.BELONGS_TO,
                    coverage_id,
                    "project:root"
                )
            })
            
            # Process package coverage
            packages = root.find("packages")
            if packages is not None:
                for package in packages.findall("package"):
                    package_name = package.get("name", "")
                    package_line_rate = float(package.get("line-rate", 0.0))
                    
                    # Process classes (files) in package
                    classes = package.find("classes")
                    if classes is not None:
                        for cls in classes.findall("class"):
                            filename = cls.get("filename", "")
                            if not filename:
                                continue
                            
                            cls_line_rate = float(cls.get("line-rate", 0.0))
                            cls_id = f"coverage_file:{abs(hash(filename))}"
                            
                            events.append({
                                "event_type": "node_update",
                                "node": create_node(
                                    NodeType.TEST,
                                    f"Coverage: {Path(filename).name}",
                                    node_id=cls_id,
                                    content={
                                        "type": "file_coverage",
                                        "filename": filename,
                                        "package": package_name,
                                        "line_coverage": cls_line_rate * 100,
                                        "covered_lines": 0,  # Would need to count from lines
                                        "total_lines": 0,
                                        "tool": "coverage.py"
                                    }
                                )
                            })
                            
                            # Link file coverage to overall coverage
                            events.append({
                                "event_type": "edge_create",
                                "edge": create_edge(
                                    RelationType.BELONGS_TO,
                                    cls_id,
                                    coverage_id
                                )
                            })
                            
                            # Try to link to corresponding source file
                            # This would create VERIFIES relationship
                            source_file_id = f"codemodule:{abs(hash(filename))}"
                            events.append({
                                "event_type": "edge_create",
                                "edge": create_edge(
                                    RelationType.VERIFIES,
                                    cls_id,
                                    source_file_id
                                )
                            })
        
        except ET.ParseError as e:
            print(f"⚠️  Invalid coverage XML in {xml_file}: {e}")
        
        return events


class TestResultSensor(SensorInterface):
    """Generic test result sensor for various formats"""
    
    def __init__(self, project_root: Path):
        super().__init__(name="test_results", description="Generic test result monitoring")
        self.project_root = project_root
    
    async def sense(self) -> List[Dict[str, Any]]:
        """Sense various test result formats"""
        events = []
        
        # Look for pytest-json-report output
        json_reports = list(self.project_root.glob("**/pytest-report.json"))
        json_reports.extend(list(self.project_root.glob("**/test-report.json")))
        
        for report_file in json_reports:
            if not report_file.is_file():
                continue
            
            # Check if file is newer than last scan
            if report_file.stat().st_mtime <= self.last_scan_time:
                continue
            
            try:
                events.extend(self._parse_pytest_json(report_file))
            except Exception as e:
                print(f"⚠️  Failed to parse pytest JSON {report_file}: {e}")
        
        if events:
            self.update_watermark()
        
        return events
    
    def _parse_pytest_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Parse pytest JSON report"""
        events = []
        
        try:
            with open(json_file, 'r') as f:
                report = json.load(f)
            
            # Create test run node
            run_id = f"testrun:{abs(hash(str(json_file)))}"
            
            summary = report.get("summary", {})
            
            events.append({
                "event_type": "node_update",
                "node": create_node(
                    NodeType.TESTSUITE,
                    f"Pytest Run",
                    node_id=run_id,
                    content={
                        "total": summary.get("total", 0),
                        "passed": summary.get("passed", 0),
                        "failed": summary.get("failed", 0),
                        "skipped": summary.get("skipped", 0),
                        "error": summary.get("error", 0),
                        "duration": report.get("duration", 0.0),
                        "start_time": report.get("start_timestamp"),
                        "test_framework": "pytest",
                        "source_file": str(json_file)
                    }
                )
            })
            
            # Link to project
            events.append({
                "event_type": "edge_create",
                "edge": create_edge(
                    RelationType.BELONGS_TO,
                    run_id,
                    "project:root"
                )
            })
            
            # Process individual test results
            tests = report.get("tests", [])
            for test in tests:
                test_id = f"test:{abs(hash(test.get('nodeid', '')))}"
                
                events.append({
                    "event_type": "node_update",
                    "node": create_node(
                        NodeType.TEST,
                        test.get("nodeid", "Unknown Test"),
                        node_id=test_id,
                        content={
                            "test_name": test.get("nodeid"),
                            "status": test.get("outcome"),
                            "duration": test.get("duration", 0.0),
                            "test_framework": "pytest",
                            "file": test.get("file"),
                            "line": test.get("line")
                        }
                    )
                })
                
                # Link to test run
                events.append({
                    "event_type": "edge_create",
                    "edge": create_edge(
                        RelationType.BELONGS_TO,
                        test_id,
                        run_id
                    )
                })
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Invalid pytest JSON in {json_file}: {e}")
        
        return events