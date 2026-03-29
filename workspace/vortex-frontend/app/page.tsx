'use client';

import { useEffect, useState } from 'react';
import { 
  Terminal, 
  Cpu, 
  Code, 
  Zap, 
  Shield, 
  Layers, 
  Search, 
  Database,
  Bot,
  ArrowRight,
  Play,
  FileCode,
  Tool,
  RefreshCw,
  Save,
  Plug
} from 'lucide-react';

const features = [
  {
    icon: Terminal,
    title: 'Rich TUI',
    description: 'Custom terminal interface with live streaming output, structured tool panels, and approval prompts.'
  },
  {
    icon: Code,
    title: 'Code Tools',
    description: 'Built-in tools for reading, writing, editing files, searching code, and running shell commands.'
  },
  {
    icon: Search,
    title: 'Codebase Index',
    description: 'Lightweight symbol extraction for Python, C, C++, JS, TS, Go, Rust, and Java.'
  },
  {
    icon: Database,
    title: 'Workspace Snapshot',
    description: 'Compact project context injected into every session for better agent awareness.'
  },
  {
    icon: Shield,
    title: 'Approval Policies',
    description: 'Configurable safety modes from fully automatic to manual approval for risky actions.'
  },
  {
    icon: Zap,
    title: 'Multi-Provider',
    description: 'Support for OpenAI, OpenRouter, and any OpenAI-compatible API with custom model profiles.'
  },
  {
    icon: Save,
    title: 'Sessions & Checkpoints',
    description: 'Save, resume, and restore conversation state with named checkpoints.'
  },
  {
    icon: Plug,
    title: 'MCP Support',
    description: 'Connect to Model Context Protocol servers and register their tools dynamically.'
  },
  {
    icon: Tool,
    title: 'Custom Tools',
    description: 'Drop Python files into .ai-agent/tools/ and they are auto-discovered and registered.'
  },
  {
    icon: RefreshCw,
    title: 'Hooks System',
    description: 'Execute custom logic before/after agent runs and tool execution.'
  },
  {
    icon: Bot,
    title: 'Sub-Agents',
    title: 'Sub-Agents',
    description: 'Built-in codebase investigator and code reviewer agents for complex tasks.'
  },
  {
    icon: FileCode,
    title: 'Memory System',
    description: 'Persistent user memory loaded from app data and injected into context.'
  }
];

const commands = [
  '/help', '/exit', '/quit', '/clear', '/scan', '/index',
  '/cwd [path|index]', '/recent', '/config', '/models [refresh]',
  '/model <name|number>', '/approval <mode>', '/stats', '/tools',
  '/mcp', '/save', '/checkpoint [name]', '/sessions',
  '/resume <session_id>', '/restore <checkpoint_id>'
];

const codeExamples = [
  {
    label: 'Single Prompt Mode',
    code: `python3 main.py "write a hello world program in c"`
  },
  {
    label: 'Interactive Mode',
    code: `python3 main.py

╭─ you › /scan
└─ Scanning workspace...

╭─ you › write a REST API in Python
└─ [VORTEX streams response...]`
  },
  {
    label: 'Docker Deployment',
    code: `docker run --rm -it \\
  --env-file .env \\
  -v "$PWD":/workspace \\
  -v vortex-data:/data \\
  vortex`
  }
];

export default function Home() {
  const [activeTab, setActiveTab] = useState(0);
  const [typedText, setTypedText] = useState('');
  const fullText = '> Analyzing codebase structure...';
  
  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index <= fullText.length) {
        setTypedText(fullText.slice(0, index));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-[#ededed]">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-cyan-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
              <Cpu className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">VORTEX</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-gray-400 hover:text-cyan-400 transition">Features</a>
            <a href="#demo" className="text-gray-400 hover:text-cyan-400 transition">Demo</a>
            <a href="#docs" className="text-gray-400 hover:text-cyan-400 transition">Docs</a>
            <button className="btn-primary text-sm">Get Started</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/30 mb-8">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="text-sm text-cyan-400">Local Terminal Coding Agent</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            Your AI Pair Programmer<br />
            <span className="gradient-text">In The Terminal</span>
          </h1>
          
          <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-10">
            VORTEX is a local terminal coding agent built around an OpenAI-compatible chat API, 
            a Rich-based TUI, and a powerful tool system for reading files, editing code, 
            running shell commands, and managing sessions.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <button className="btn-primary flex items-center justify-center gap-2">
              <Play className="w-5 h-5" />
              Try It Now
            </button>
            <button className="btn-secondary flex items-center justify-center gap-2">
              View on GitHub
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          {/* Terminal Demo */}
          <div className="terminal-window max-w-4xl mx-auto animate-float">
            <div className="terminal-header">
              <div className="terminal-dot red"></div>
              <div className="terminal-dot yellow"></div>
              <div className="terminal-dot green"></div>
              <span className="ml-4 text-sm text-gray-400">vortex — bash — 80×24</span>
            </div>
            <div className="terminal-body">
              <div className="mb-2">
                <span className="terminal-prompt">╭─ you ›</span> scan workspace
              </div>
              <div className="text-gray-500 mb-2">
                {typedText}<span className="terminal-cursor"></span>
              </div>
              <div className="text-cyan-400 mb-2">
                ✓ Workspace snapshot generated (12 files, 3.2KB)
              </div>
              <div className="text-cyan-400 mb-2">
                ✓ Symbol index built (24 functions, 8 classes)
              </div>
              <div className="mb-2">
                <span className="terminal-prompt">╭─ you ›</span> write a function to sort an array
              </div>
              <div className="text-green-400">
                <span className="text-gray-500">[VORTEX is thinking...]</span><br />
                Here's a quicksort implementation in Python:
                <br /><br />
                <span className="text-yellow-300">def</span> <span className="text-blue-400">quicksort</span>(arr):<br />
                &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-purple-400">if</span> len(arr) &lt;= <span className="text-orange-400">1</span>:<br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-purple-400">return</span> arr<br />
                &nbsp;&nbsp;&nbsp;&nbsp;pivot = arr[len(arr) // <span className="text-orange-400">2</span>]<br />
                &nbsp;&nbsp;&nbsp;&nbsp;left = [x <span className="text-purple-400">for</span> x <span className="text-purple-400">in</span> arr <span className="text-purple-400">if</span> x &lt; pivot]<br />
                &nbsp;&nbsp;&nbsp;&nbsp;middle = [x <span className="text-purple-400">for</span> x <span className="text-purple-400">in</span> arr <span className="text-purple-400">if</span> x == pivot]<br />
                &nbsp;&nbsp;&nbsp;&nbsp;right = [x <span className="text-purple-400">for</span> x <span className="text-purple-400">in</span> arr <span className="text-purple-400">if</span> x &gt; pivot]<br />
                &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-purple-400">return</span> quicksort(left) + middle + quicksort(right)
              </div>
              <div className="mt-4">
                <span className="terminal-prompt">╰─ you ›</span><span className="terminal-cursor"></span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="py-20 px-6 bg-[#0d0d0d]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="gradient-text">Powerful Features</span>
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Everything you need for AI-assisted coding in a local, secure terminal environment.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="feature-card p-6 rounded-xl bg-[#1a1a1a] border border-gray-800"
              >
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-600/20 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-cyan-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Code Examples */}
      <section id="demo" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="gradient-text">See It In Action</span>
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Run VORTEX in different modes to suit your workflow.
            </p>
          </div>
          
          <div className="terminal-window max-w-4xl mx-auto">
            <div className="terminal-header">
              <div className="terminal-dot red"></div>
              <div className="terminal-dot yellow"></div>
              <div className="terminal-dot green"></div>
              <div className="ml-auto flex gap-2">
                {codeExamples.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setActiveTab(index)}
                    className={`px-3 py-1 rounded text-sm transition ${
                      activeTab === index 
                        ? 'bg-cyan-500/30 text-cyan-400' 
                        : 'text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="terminal-body">
              <pre className="font-mono text-sm whitespace-pre-wrap">
                <code className="text-gray-300">
                  {codeExamples[activeTab].code}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Commands Section */}
      <section id="docs" className="py-20 px-6 bg-[#0d0d0d]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              <span className="gradient-text">Interactive Commands</span>
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Control VORTEX with intuitive slash commands.
            </p>
          </div>
          
          <div className="flex flex-wrap justify-center gap-3 max-w-4xl mx-auto">
            {commands.map((cmd, index) => (
              <div 
                key={index}
                className="px-4 py-2 rounded-lg bg-[#1a1a1a] border border-gray-800 text-cyan-400 font-mono text-sm hover:border-cyan-500/50 transition cursor-default"
              >
                {cmd}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="p-12 rounded-2xl bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-cyan-500/10 border border-cyan-500/30">
            <h2 className="text-4xl font-bold mb-4">Ready to Get Started?</h2>
            <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
              Install VORTEX and start coding with your AI assistant today. 
              Works with any OpenAI-compatible API.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="btn-primary">
                Install VORTEX
              </button>
              <button className="btn-secondary">
                Read Documentation
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-800">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold">VORTEX</span>
          </div>
          <div className="text-gray-500 text-sm">
            Built with ❤️ for developers who love the terminal
          </div>
          <div className="flex gap-6">
            <a href="#" className="text-gray-500 hover:text-cyan-400 transition">GitHub</a>
            <a href="#" className="text-gray-500 hover:text-cyan-400 transition">Docs</a>
            <a href="#" className="text-gray-500 hover:text-cyan-400 transition">Discord</a>
          </div>
        </div>
      </footer>
    </main>
  );
}