#!/usr/bin/env python3
"""
Comprehensive Test Suite for AO RAG API with Llama 3.2 1B Integration
This script tests all API endpoints and verifies Llama 3.2 1B model integration
"""

import requests
import json
import time
import sys
from datetime import datetime


class AORAGAPITester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def log_test(self, test_name, success, message="", response_data=None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {message}")

    def test_health_check(self):
        """Test the health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and data.get("system_initialized"):
                    self.log_test("Health Check", True,
                                  "API is healthy and initialized", data)
                    return True
                else:
                    self.log_test("Health Check", False,
                                  "System not properly initialized")
                    return False
            else:
                self.log_test("Health Check", False,
                              f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Connection error: {e}")
            return False

    def test_system_stats(self):
        """Test the system statistics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/stats")

            if response.status_code == 200:
                data = response.json()
                stats = data.get("statistics", {})

                if stats.get("total_aos", 0) > 0:
                    self.log_test("System Stats", True,
                                  f"Found {stats['total_aos']} AOs, avg risk: {stats.get('avg_risk_score', 0)}",
                                  stats)
                    return True
                else:
                    self.log_test("System Stats", False,
                                  "No AOs found in system")
                    return False
            else:
                self.log_test("System Stats", False,
                              f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("System Stats", False, f"Error: {e}")
            return False

    def test_search_functionality(self):
        """Test the search endpoint with AI analysis"""
        test_queries = [
            "high risk applications",
            "compliance issues",
            "security vulnerabilities",
            "Alice Singh"
        ]

        success_count = 0

        for query in test_queries:
            try:
                payload = {"query": query, "top_k": 3}
                response = self.session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Check if we got search results
                    matching_aos = data.get("matching_aos", [])
                    ai_analysis = data.get("ai_analysis", {})

                    if matching_aos or ai_analysis:
                        success_count += 1
                        self.log_test(f"Search Query: '{query}'", True,
                                      f"Found {len(matching_aos)} matches with AI analysis")

                        # Check for AI-powered analysis
                        if ai_analysis.get("search_summary"):
                            print(
                                f"   📊 AI Summary: {ai_analysis['search_summary'][:100]}...")

                    else:
                        self.log_test(
                            f"Search Query: '{query}'", False, "No results or AI analysis")
                else:
                    self.log_test(
                        f"Search Query: '{query}'", False, f"HTTP {response.status_code}")

            except Exception as e:
                self.log_test(f"Search Query: '{query}'", False, f"Error: {e}")

        overall_success = success_count == len(test_queries)
        self.log_test("Search Functionality Overall", overall_success,
                      f"{success_count}/{len(test_queries)} queries successful")
        return overall_success

    def test_suggestions_with_llm(self):
        """Test the suggestions endpoint with LLM integration"""
        test_aos = ["Alice Singh", "Bob Johnson", "Charlie Brown"]

        success_count = 0

        for ao_name in test_aos:
            try:
                payload = {"ao_name": ao_name, "use_llm": True}
                response = self.session.post(
                    f"{self.base_url}/suggestions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get("suggestions", {})

                    if suggestions.get("status") == "ao_found":
                        ai_analysis = suggestions.get("ai_analysis", {})

                        if ai_analysis.get("executive_summary"):
                            success_count += 1
                            self.log_test(f"Suggestions for '{ao_name}'", True,
                                          "LLM-powered analysis generated successfully")

                            # Show AI-generated content samples
                            if ai_analysis.get("executive_summary"):
                                print(
                                    f"   🤖 AI Summary: {ai_analysis['executive_summary'][:100]}...")
                            if ai_analysis.get("immediate_actions"):
                                print(
                                    f"   ⚡ Actions: {len(ai_analysis['immediate_actions'])} immediate actions")
                        else:
                            self.log_test(
                                f"Suggestions for '{ao_name}'", False, "No AI analysis generated")

                    elif suggestions.get("status") == "ao_not_found":
                        # This is expected for some test names
                        similar_aos = suggestions.get("similar_ao_names", [])
                        if similar_aos:
                            print(f"   💡 Similar AOs found: {similar_aos[:3]}")
                        self.log_test(
                            f"Suggestions for '{ao_name}'", True, "AO not found (expected)")
                        success_count += 1
                    else:
                        self.log_test(
                            f"Suggestions for '{ao_name}'", False, "Unexpected response format")
                else:
                    self.log_test(
                        f"Suggestions for '{ao_name}'", False, f"HTTP {response.status_code}")

            except Exception as e:
                self.log_test(
                    f"Suggestions for '{ao_name}'", False, f"Error: {e}")

        overall_success = success_count >= len(
            test_aos) * 0.7  # Allow 30% failure for test names
        self.log_test("Suggestions with LLM Overall", overall_success,
                      f"{success_count}/{len(test_aos)} tests successful")
        return overall_success

    def test_llama_integration(self):
        """Test specific Llama 3.2 1B integration"""
        try:
            # Test with a specific query that should trigger detailed AI analysis
            payload = {"query": "critical vulnerabilities security analysis"}
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                ai_analysis = data.get("ai_analysis", {})

                # Check for signs of detailed AI analysis
                llm_indicators = [
                    ai_analysis.get("search_summary", ""),
                    ai_analysis.get("risk_analysis", ""),
                    ai_analysis.get("comparative_insights", "")
                ]

                # Check if the response contains substantial AI-generated content
                total_content = sum(len(str(indicator))
                                    for indicator in llm_indicators)

                if total_content > 200:  # Substantial AI content
                    self.log_test("Llama 3.2 1B Integration", True,
                                  f"AI generated {total_content} characters of analysis")
                    return True
                else:
                    self.log_test("Llama 3.2 1B Integration", False,
                                  "Insufficient AI content generated")
                    return False
            else:
                self.log_test("Llama 3.2 1B Integration", False,
                              f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Llama 3.2 1B Integration", False, f"Error: {e}")
            return False

    def test_error_handling(self):
        """Test API error handling"""
        error_tests = [
            ("Empty search query", "POST", "/search", {"query": ""}),
            ("Missing AO name", "POST", "/suggestions", {"use_llm": True}),
            ("Invalid endpoint", "GET", "/invalid", {}),
            ("Malformed JSON", "POST", "/search", "invalid json")
        ]

        success_count = 0

        for test_name, method, endpoint, payload in error_tests:
            try:
                url = f"{self.base_url}{endpoint}"

                if method == "GET":
                    response = self.session.get(url)
                else:
                    response = self.session.post(url, json=payload if payload != "invalid json" else None,
                                                 data=payload if payload == "invalid json" else None,
                                                 headers={"Content-Type": "application/json"})

                # We expect 4xx status codes for these error tests
                if 400 <= response.status_code < 500:
                    success_count += 1
                    self.log_test(f"Error Handling: {test_name}", True,
                                  f"Properly returned HTTP {response.status_code}")
                else:
                    self.log_test(f"Error Handling: {test_name}", False,
                                  f"Unexpected status code: {response.status_code}")

            except Exception as e:
                self.log_test(
                    f"Error Handling: {test_name}", False, f"Exception: {e}")

        overall_success = success_count >= len(error_tests) * 0.75
        self.log_test("Error Handling Overall", overall_success,
                      f"{success_count}/{len(error_tests)} error tests passed")
        return overall_success

    def run_comprehensive_test(self):
        """Run all tests and generate report"""
        print("🚀 Starting Comprehensive AO RAG API Test Suite")
        print("=" * 60)

        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        time.sleep(2)

        test_functions = [
            self.test_health_check,
            self.test_system_stats,
            self.test_search_functionality,
            self.test_suggestions_with_llm,
            self.test_llama_integration,
            self.test_error_handling
        ]

        passed_tests = 0
        total_tests = len(test_functions)

        for test_func in test_functions:
            print(
                f"\n📋 Running {test_func.__name__.replace('test_', '').replace('_', ' ').title()}...")
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test function failed: {e}")

        # Generate final report
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY REPORT")
        print("=" * 60)

        success_rate = (passed_tests / total_tests) * 100
        print(
            f"Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} test suites passed)")

        if success_rate >= 80:
            print("🎉 EXCELLENT! API is production-ready with Llama 3.2 1B integration")
        elif success_rate >= 60:
            print("⚠️  GOOD! Most functionality working, minor issues to address")
        else:
            print("🔧 NEEDS WORK! Several critical issues need fixing")

        # Detailed test results
        print(
            f"\nDetailed Results ({len(self.test_results)} individual tests):")
        passed_individual = sum(
            1 for result in self.test_results if result["success"])
        print(
            f"Individual Test Success Rate: {(passed_individual/len(self.test_results)*100):.1f}%")

        # Show any failures
        failures = [
            result for result in self.test_results if not result["success"]]
        if failures:
            print(f"\n❌ Failed Tests ({len(failures)}):")
            for failure in failures:
                print(f"   - {failure['test']}: {failure['message']}")

        print(
            f"\n✅ API Endpoints Working: {4 - len([f for f in failures if 'HTTP' in f.get('message', '')])}/4")
        print(
            f"🤖 Llama 3.2 1B Integration: {'✅ Working' if any('Llama' in r['test'] and r['success'] for r in self.test_results) else '❌ Issues'}")

        return success_rate >= 80


if __name__ == "__main__":
    print("🔍 AO RAG API Test Suite - Llama 3.2 1B Integration")
    print("Testing all endpoints and AI functionality...")

    tester = AORAGAPITester()

    # Check if server is running
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code != 200:
            print(
                "❌ Server is not responding properly. Please start the API server first.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(
            "❌ Cannot connect to API server. Please start it with: python minimal_ao_api.py")
        sys.exit(1)

    # Run comprehensive tests
    success = tester.run_comprehensive_test()

    if success:
        print("\n🎯 API is ready for production use!")
        print("📋 You can now test with Postman using the Enhanced_AO_API_Postman_Collection.json")
    else:
        print("\n🔧 Some issues found. Please review the failed tests above.")

    sys.exit(0 if success else 1)
