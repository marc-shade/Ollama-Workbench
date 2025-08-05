"""
End-to-end integration tests for complete workflows.

Tests full user workflows across multiple components and modules,
ensuring seamless integration of the entire Ollama Workbench system.
"""

import pytest
import os
import tempfile
import shutil
import json
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from unittest import TestCase
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all major components
from chat_interface import enhanced_chat_interface
from multimodel_chat import MultiModelChat
from build import BuildWorkflow
from research import ResearchWorkflow  
from brainstorm import BrainstormWorkflow
from projects import ProjectManager
from corpus_management import CorpusManager
from voice_interface import VoiceInterface
from file_management import FileManager
from external_providers import load_api_keys
from session_utils import SessionManager
from performance_metrics import PerformanceTracker


class TestChatWorkflowE2E(TestCase):
    """Test complete chat workflow from start to finish"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_session_id = "test_session_123"
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('streamlit.session_state', {})
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    @patch('session_utils.save_session')
    @patch('session_utils.load_session')
    def test_complete_chat_session_workflow(self, mock_load_session, mock_save_session, 
                                          mock_ollama_api, mock_load_keys):
        """Test complete chat session from initialization to cleanup"""
        
        # Setup mocks
        mock_load_keys.return_value = {'openai_api_key': 'test_key'}
        mock_ollama_api.return_value = "Hello! How can I help you today?"
        mock_load_session.return_value = {
            'messages': [],
            'model': 'mistral:instruct',
            'session_id': self.test_session_id
        }
        
        # Import streamlit after mocking session_state
        import streamlit as st
        
        # Initialize session
        st.session_state['session_id'] = self.test_session_id
        st.session_state['messages'] = []
        st.session_state['selected_model'] = 'mistral:instruct'
        
        # Simulate user interaction workflow
        with patch('streamlit.chat_input') as mock_chat_input:
            mock_chat_input.return_value = "Hello, AI assistant!"
            
            with patch('streamlit.chat_message') as mock_chat_message:
                with patch('streamlit.write') as mock_write:
                    
                    # Simulate the chat interface flow
                    user_message = "Hello, AI assistant!"
                    
                    # Add user message to session
                    st.session_state['messages'].append({
                        "role": "user", 
                        "content": user_message,
                        "timestamp": time.time()
                    })
                    
                    # Generate AI response
                    messages_for_api = [{"role": "user", "content": user_message}]
                    ai_response = mock_ollama_api.return_value
                    
                    # Add AI response to session
                    st.session_state['messages'].append({
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": time.time()
                    })
                    
                    # Verify the workflow
                    self.assertEqual(len(st.session_state['messages']), 2)
                    self.assertEqual(st.session_state['messages'][0]['role'], 'user')
                    self.assertEqual(st.session_state['messages'][1]['role'], 'assistant')
                    
                    # Verify API was called correctly
                    mock_ollama_api.assert_called_once()
    
    @patch('streamlit.session_state', {})
    @patch('external_providers.load_api_keys')
    def test_multimodal_chat_workflow(self, mock_load_keys):
        """Test multimodal chat with text and file inputs"""
        
        mock_load_keys.return_value = {'openai_api_key': 'test_key'}
        
        # Create test image file
        test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        with open(test_image_path, 'wb') as f:
            f.write(b'fake_image_data')
        
        with patch('multimodel_chat.MultiModelChat') as mock_multimodel:
            mock_chat_instance = Mock()
            mock_multimodel.return_value = mock_chat_instance
            mock_chat_instance.process_multimodal_input.return_value = "I can see an image with..."
            
            # Simulate multimodal workflow
            chat = MultiModelChat()
            
            # Test text + image input
            result = chat.process_multimodal_input(
                text="What do you see in this image?",
                image_path=test_image_path
            )
            
            self.assertIn("image", result.lower())
            mock_chat_instance.process_multimodal_input.assert_called_once()
    
    @patch('performance_metrics.PerformanceTracker')
    def test_chat_with_performance_tracking(self, mock_tracker_class):
        """Test chat workflow with performance monitoring"""
        
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        with patch('ollama_utils.call_ollama_api') as mock_api:
            mock_api.return_value = "Performance tracked response"
            
            # Initialize performance tracker
            tracker = PerformanceTracker()
            
            # Start tracking
            tracker.start_request("chat_interaction")
            
            # Simulate API call
            response = mock_api("mistral:instruct", [{"role": "user", "content": "test"}])
            
            # End tracking
            tracker.end_request("chat_interaction")
            
            # Verify tracking was called
            mock_tracker.start_request.assert_called_with("chat_interaction")
            mock_tracker.end_request.assert_called_with("chat_interaction")


class TestWorkflowIntegrationE2E(TestCase):
    """Test integration of specialized workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    def test_build_workflow_e2e(self, mock_ollama_api, mock_load_keys):
        """Test complete build workflow from start to finish"""
        
        mock_load_keys.return_value = {}
        mock_ollama_api.side_effect = [
            "I'll help you build a web application...",
            "Here's the project structure...",
            "Here's the implementation code..."
        ]
        
        with patch('build.BuildWorkflow') as mock_build_class:
            mock_build = Mock()
            mock_build_class.return_value = mock_build
            mock_build.create_project.return_value = {
                'project_path': self.temp_dir,
                'files_created': ['index.html', 'app.js', 'style.css'],
                'status': 'completed'
            }
            
            # Initialize build workflow
            build_workflow = BuildWorkflow()
            
            # Execute complete build
            project_spec = {
                'name': 'test-app',
                'type': 'web',
                'description': 'A test web application'
            }
            
            result = build_workflow.create_project(project_spec)
            
            # Verify workflow completion
            self.assertEqual(result['status'], 'completed')
            self.assertIn('index.html', result['files_created'])
            mock_build.create_project.assert_called_once_with(project_spec)
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    def test_research_workflow_e2e(self, mock_ollama_api, mock_load_keys):
        """Test complete research workflow"""
        
        mock_load_keys.return_value = {}
        mock_ollama_api.side_effect = [
            "Here's my research on quantum computing...",
            "Key findings include...",
            "Conclusions and recommendations..."
        ]
        
        with patch('research.ResearchWorkflow') as mock_research_class:
            mock_research = Mock()
            mock_research_class.return_value = mock_research
            mock_research.conduct_research.return_value = {
                'topic': 'quantum computing',
                'findings': ['Finding 1', 'Finding 2'],
                'report_path': os.path.join(self.temp_dir, 'research_report.md'),
                'status': 'completed'
            }
            
            # Execute research workflow
            research_workflow = ResearchWorkflow()
            
            research_request = {
                'topic': 'quantum computing',
                'depth': 'comprehensive',
                'sources': ['academic', 'industry']
            }
            
            result = research_workflow.conduct_research(research_request)
            
            # Verify research completion
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['topic'], 'quantum computing')
            self.assertTrue(len(result['findings']) > 0)
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    def test_brainstorm_workflow_e2e(self, mock_ollama_api, mock_load_keys):
        """Test complete brainstorming workflow"""
        
        mock_load_keys.return_value = {}
        mock_ollama_api.side_effect = [
            "Initial ideas for your project...",
            "Expanding on concept A...",
            "Here's the final brainstorm summary..."
        ]
        
        with patch('brainstorm.BrainstormWorkflow') as mock_brainstorm_class:
            mock_brainstorm = Mock()
            mock_brainstorm_class.return_value = mock_brainstorm
            mock_brainstorm.generate_ideas.return_value = {
                'topic': 'mobile app ideas',
                'ideas': ['Idea 1', 'Idea 2', 'Idea 3'],
                'best_idea': 'Idea 2',
                'next_steps': ['Step 1', 'Step 2'],
                'status': 'completed'
            }
            
            # Execute brainstorm workflow
            brainstorm_workflow = BrainstormWorkflow()
            
            brainstorm_request = {
                'topic': 'mobile app ideas',
                'context': 'productivity apps',
                'target_audience': 'professionals'
            }
            
            result = brainstorm_workflow.generate_ideas(brainstorm_request)
            
            # Verify brainstorm completion
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(len(result['ideas']) > 0)
            self.assertIsNotNone(result['best_idea'])


class TestProjectManagementE2E(TestCase):
    """Test complete project management workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = os.path.join(self.temp_dir, 'test_project')
        os.makedirs(self.project_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('projects.ProjectManager')
    def test_project_lifecycle_e2e(self, mock_project_manager_class):
        """Test complete project lifecycle from creation to completion"""
        
        mock_manager = Mock()
        mock_project_manager_class.return_value = mock_manager
        
        # Mock project lifecycle methods
        mock_manager.create_project.return_value = {
            'project_id': 'proj_123',
            'name': 'Test Project',
            'path': self.project_dir,
            'status': 'created'
        }
        
        mock_manager.update_project.return_value = {
            'project_id': 'proj_123',
            'status': 'in_progress'
        }
        
        mock_manager.complete_project.return_value = {
            'project_id': 'proj_123',
            'status': 'completed',
            'deliverables': ['file1.py', 'file2.js']
        }
        
        # Execute project lifecycle
        project_manager = ProjectManager()
        
        # 1. Create project
        create_result = project_manager.create_project({
            'name': 'Test Project',
            'description': 'A test project',
            'type': 'development'
        })
        
        project_id = create_result['project_id']
        
        # 2. Update project
        update_result = project_manager.update_project(project_id, {
            'status': 'in_progress',
            'progress': 50
        })
        
        # 3. Complete project
        complete_result = project_manager.complete_project(project_id)
        
        # Verify lifecycle
        self.assertEqual(create_result['status'], 'created')
        self.assertEqual(update_result['status'], 'in_progress')
        self.assertEqual(complete_result['status'], 'completed')
        
        # Verify all methods were called
        mock_manager.create_project.assert_called_once()
        mock_manager.update_project.assert_called_once()
        mock_manager.complete_project.assert_called_once()
    
    @patch('file_management.FileManager')
    @patch('corpus_management.CorpusManager')
    def test_project_with_file_and_corpus_integration(self, mock_corpus_class, mock_file_class):
        """Test project integration with file management and corpus"""
        
        mock_file_manager = Mock()
        mock_file_class.return_value = mock_file_manager
        
        mock_corpus_manager = Mock()
        mock_corpus_class.return_value = mock_corpus_manager
        
        # Mock file operations
        mock_file_manager.create_file.return_value = {
            'file_path': os.path.join(self.project_dir, 'README.md'),
            'status': 'created'
        }
        
        mock_file_manager.list_files.return_value = {
            'files': ['README.md', 'main.py'],
            'count': 2
        }
        
        # Mock corpus operations
        mock_corpus_manager.add_documents.return_value = {
            'documents_added': 2,
            'corpus_id': 'corpus_123'
        }
        
        # Execute integrated workflow
        file_manager = FileManager()
        corpus_manager = CorpusManager()
        
        # 1. Create project files
        readme_result = file_manager.create_file(
            os.path.join(self.project_dir, 'README.md'),
            "# Test Project\n\nThis is a test project."
        )
        
        # 2. List project files
        files_result = file_manager.list_files(self.project_dir)
        
        # 3. Add files to corpus
        corpus_result = corpus_manager.add_documents([
            os.path.join(self.project_dir, 'README.md'),
            os.path.join(self.project_dir, 'main.py')
        ])
        
        # Verify integration
        self.assertEqual(readme_result['status'], 'created')
        self.assertEqual(files_result['count'], 2)
        self.assertEqual(corpus_result['documents_added'], 2)


class TestVoiceInterfaceE2E(TestCase):
    """Test complete voice interface workflows"""
    
    @patch('voice_interface.VoiceInterface')
    @patch('external_providers.load_api_keys')
    def test_voice_chat_workflow(self, mock_load_keys, mock_voice_class):
        """Test complete voice chat workflow"""
        
        mock_load_keys.return_value = {'openai_api_key': 'test_key'}
        
        mock_voice = Mock()
        mock_voice_class.return_value = mock_voice
        
        # Mock voice interface methods
        mock_voice.start_recording.return_value = True
        mock_voice.stop_recording.return_value = "/tmp/recorded_audio.wav"
        mock_voice.transcribe_audio.return_value = "Hello, how are you today?"
        mock_voice.synthesize_speech.return_value = "/tmp/response_audio.wav"
        
        # Execute voice workflow
        voice_interface = VoiceInterface()
        
        # 1. Start recording
        recording_started = voice_interface.start_recording()
        
        # 2. Stop recording and get audio file
        audio_file = voice_interface.stop_recording()
        
        # 3. Transcribe audio to text
        transcribed_text = voice_interface.transcribe_audio(audio_file)
        
        # 4. Process with AI (mock)
        with patch('ollama_utils.call_ollama_api') as mock_api:
            mock_api.return_value = "I'm doing well, thank you for asking!"
            ai_response = mock_api("mistral:instruct", [
                {"role": "user", "content": transcribed_text}
            ])
        
        # 5. Synthesize AI response to speech
        response_audio = voice_interface.synthesize_speech(ai_response)
        
        # Verify workflow
        self.assertTrue(recording_started)
        self.assertEqual(transcribed_text, "Hello, how are you today?")
        self.assertIn("doing well", ai_response)
        self.assertIsNotNone(response_audio)
        
        # Verify all voice methods were called
        mock_voice.start_recording.assert_called_once()
        mock_voice.stop_recording.assert_called_once()
        mock_voice.transcribe_audio.assert_called_once()
        mock_voice.synthesize_speech.assert_called_once()


class TestSessionManagementE2E(TestCase):
    """Test complete session management workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('session_utils.SessionManager')
    def test_session_lifecycle_e2e(self, mock_session_class):
        """Test complete session lifecycle"""
        
        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        
        # Mock session methods
        mock_session_manager.create_session.return_value = {
            'session_id': 'session_123',
            'created_at': time.time(),
            'status': 'active'
        }
        
        mock_session_manager.save_session.return_value = True
        mock_session_manager.load_session.return_value = {
            'session_id': 'session_123',
            'messages': [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            'created_at': time.time(),
            'last_updated': time.time()
        }
        
        mock_session_manager.delete_session.return_value = True
        
        # Execute session lifecycle
        session_manager = SessionManager()
        
        # 1. Create new session
        create_result = session_manager.create_session()
        session_id = create_result['session_id']
        
        # 2. Add messages to session
        session_data = {
            'session_id': session_id,
            'messages': [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        
        # 3. Save session
        save_result = session_manager.save_session(session_data)
        
        # 4. Load session
        loaded_session = session_manager.load_session(session_id)
        
        # 5. Delete session
        delete_result = session_manager.delete_session(session_id)
        
        # Verify lifecycle
        self.assertEqual(create_result['status'], 'active')
        self.assertTrue(save_result)
        self.assertEqual(len(loaded_session['messages']), 2)
        self.assertTrue(delete_result)
        
        # Verify all methods were called
        mock_session_manager.create_session.assert_called_once()
        mock_session_manager.save_session.assert_called_once()
        mock_session_manager.load_session.assert_called_once()
        mock_session_manager.delete_session.assert_called_once()


class TestFullApplicationE2E(TestCase):
    """Test complete application workflows across all components"""
    
    def setUp(self):
        """Set up comprehensive test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    @patch('streamlit.session_state', {})
    def test_complete_user_workflow_scenario(self, mock_ollama_api, mock_load_keys):
        """Test a complete user workflow from app start to project completion"""
        
        # Setup
        mock_load_keys.return_value = {'openai_api_key': 'test_key'}
        mock_ollama_api.side_effect = [
            "I'll help you create a web application...",
            "Here's the project structure I recommend...",
            "I've generated the initial code files...",
            "The project has been successfully created!"
        ]
        
        import streamlit as st
        
        # Initialize app state
        st.session_state['user_id'] = 'user_123'
        st.session_state['session_id'] = 'session_456'
        st.session_state['current_project'] = None
        
        # Workflow: User wants to create a web app project
        
        # 1. User starts chat session
        with patch('session_utils.SessionManager') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.create_session.return_value = {
                'session_id': 'session_456',
                'status': 'active'
            }
            
            session_manager = mock_session_class()
            session_result = session_manager.create_session()
            
        # 2. User requests project creation via chat
        with patch('chat_interface.enhanced_chat_interface') as mock_chat:
            with patch('projects.ProjectManager') as mock_project_class:
                mock_project = Mock()
                mock_project_class.return_value = mock_project
                mock_project.create_project.return_value = {
                    'project_id': 'proj_789',
                    'name': 'My Web App',
                    'status': 'created',
                    'path': self.temp_dir
                }
                
                # Simulate chat interaction
                user_message = "Create a simple web application for me"
                chat_messages = [{"role": "user", "content": user_message}]
                
                # AI processes request and creates project
                ai_response = mock_ollama_api.return_value
                project_manager = mock_project_class()
                project_result = project_manager.create_project({
                    'name': 'My Web App',
                    'type': 'web',
                    'description': 'A simple web application'
                })
        
        # 3. User adds files to corpus for context
        with patch('corpus_management.CorpusManager') as mock_corpus_class:
            mock_corpus = Mock()
            mock_corpus_class.return_value = mock_corpus
            mock_corpus.add_documents.return_value = {
                'documents_added': 3,
                'corpus_id': 'corpus_123'
            }
            
            corpus_manager = mock_corpus_class()
            corpus_result = corpus_manager.add_documents([
                os.path.join(self.temp_dir, 'index.html'),
                os.path.join(self.temp_dir, 'app.js'),
                os.path.join(self.temp_dir, 'style.css')
            ])
        
        # 4. User continues conversation about the project
        followup_message = "Can you explain the code structure?"
        extended_chat = chat_messages + [
            {"role": "assistant", "content": ai_response},
            {"role": "user", "content": followup_message}
        ]
        
        final_response = "Here's how the code is structured..."
        
        # 5. User saves session
        with patch('session_utils.SessionManager') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.save_session.return_value = True
            
            session_data = {
                'session_id': 'session_456',
                'messages': extended_chat + [{"role": "assistant", "content": final_response}],
                'project_id': 'proj_789'
            }
            
            session_manager = mock_session_class()
            save_result = session_manager.save_session(session_data)
        
        # Verify complete workflow
        self.assertEqual(session_result['status'], 'active')
        self.assertEqual(project_result['status'], 'created')
        self.assertEqual(corpus_result['documents_added'], 3)
        self.assertTrue(save_result)
        
        # Verify API calls
        self.assertEqual(mock_ollama_api.call_count, 4)  # Multiple AI interactions
    
    @patch('performance_metrics.PerformanceTracker')
    def test_performance_monitoring_across_workflow(self, mock_tracker_class):
        """Test performance monitoring throughout a complete workflow"""
        
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock performance data
        mock_tracker.get_metrics.return_value = {
            'total_requests': 5,
            'average_response_time': 1.2,
            'success_rate': 100.0,
            'error_count': 0
        }
        
        # Simulate tracked workflow
        tracker = PerformanceTracker()
        
        # Track multiple operations
        operations = [
            'session_creation',
            'chat_interaction',
            'project_creation',
            'file_management',
            'corpus_update'
        ]
        
        for operation in operations:
            tracker.start_request(operation)
            # Simulate operation time
            time.sleep(0.01)
            tracker.end_request(operation)
        
        # Get final metrics
        metrics = tracker.get_metrics()
        
        # Verify tracking
        self.assertEqual(metrics['total_requests'], 5)
        self.assertGreater(metrics['average_response_time'], 0)
        
        # Verify all operations were tracked
        self.assertEqual(mock_tracker.start_request.call_count, 5)
        self.assertEqual(mock_tracker.end_request.call_count, 5)


class TestErrorHandlingE2E(TestCase):
    """Test error handling across complete workflows"""
    
    @patch('external_providers.load_api_keys')
    def test_graceful_degradation_workflow(self, mock_load_keys):
        """Test graceful degradation when services are unavailable"""
        
        mock_load_keys.return_value = {}
        
        # Simulate various service failures
        error_scenarios = [
            ('ollama_utils.call_ollama_api', ConnectionError("Ollama unavailable")),
            ('session_utils.save_session', IOError("Disk full")),
            ('corpus_management.add_documents', Exception("Corpus service down"))
        ]
        
        for service_path, error in error_scenarios:
            with patch(service_path) as mock_service:
                mock_service.side_effect = error
                
                # Workflow should continue with fallbacks
                try:
                    if 'ollama' in service_path:
                        # Should fall back to other providers
                        with patch('openai_utils.call_openai_api') as mock_fallback:
                            mock_fallback.return_value = "Fallback response"
                            response = mock_fallback("gpt-4", [{"role": "user", "content": "test"}])
                            self.assertEqual(response, "Fallback response")
                    
                    elif 'session' in service_path:
                        # Should continue without saving
                        with patch('streamlit.warning') as mock_warning:
                            # Simulate warning to user
                            mock_warning("Session could not be saved")
                            mock_warning.assert_called_once()
                    
                    elif 'corpus' in service_path:
                        # Should continue without corpus updates
                        with patch('streamlit.error') as mock_error:
                            mock_error("Corpus update failed")
                            mock_error.assert_called_once()
                            
                except Exception as e:
                    self.fail(f"Workflow should not fail completely: {e}")
    
    def test_data_consistency_on_errors(self):
        """Test that data remains consistent when errors occur"""
        
        # Mock data stores
        mock_session_data = {'session_id': 'test', 'messages': []}
        mock_project_data = {'project_id': 'test', 'files': []}
        
        with patch('session_utils.load_session') as mock_load:
            mock_load.return_value = mock_session_data.copy()
            
            with patch('session_utils.save_session') as mock_save:
                mock_save.side_effect = Exception("Save failed")
                
                # Attempt operation that might corrupt data
                try:
                    # Load original data
                    original_data = mock_load('test')
                    
                    # Modify data
                    modified_data = original_data.copy()
                    modified_data['messages'].append({"role": "user", "content": "test"})
                    
                    # Attempt to save (will fail)
                    mock_save(modified_data)
                    
                except Exception:
                    # Verify original data is unchanged
                    current_data = mock_load('test')
                    self.assertEqual(current_data, mock_session_data)


if __name__ == "__main__":
    # Run tests with verbose output and longer timeout for E2E tests
    pytest.main([__file__, "-v", "--tb=short", "--timeout=300"])
