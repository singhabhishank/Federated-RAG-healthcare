import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import {
  Shield,
  Database,
  FileText,
  Cpu,
  ArrowRight,
  Lock,
  Activity,
  GitMerge,
  CheckCircle } from
'lucide-react';
export function LandingPage() {
  const fadeIn = {
    hidden: {
      opacity: 0,
      y: 20
    },
    visible: {
      opacity: 1,
      y: 0
    }
  };
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed w-full bg-white/80 backdrop-blur-md z-50 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">
                FederatedMed
              </span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <a
                href="#features"
                className="text-gray-600 hover:text-indigo-600 font-medium">

                Features
              </a>
              <a
                href="#how-it-works"
                className="text-gray-600 hover:text-indigo-600 font-medium">

                How it Works
              </a>
              <a
                href="#security"
                className="text-gray-600 hover:text-indigo-600 font-medium">

                Security
              </a>
              <div className="flex items-center space-x-4">
                <Link to="/login">
                  <Button variant="ghost">Sign In</Button>
                </Link>
                <Link to="/signup">
                  <Button>Get Started</Button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            transition={{
              duration: 0.5
            }}>

            <div className="inline-flex items-center px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-sm font-medium mb-6">
              <span className="flex h-2 w-2 rounded-full bg-indigo-600 mr-2"></span>
              New: Multi-Provider LLM Support
            </div>
            <h1 className="text-5xl sm:text-6xl font-bold text-gray-900 leading-tight mb-6">
              Privacy-Preserving{' '}
              <span className="text-indigo-600">Federated RAG</span> for
              Healthcare
            </h1>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Collaborative medical Q&A across hospitals without sharing raw
              patient data. Secure, compliant, and powered by differential
              privacy.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link to="/signup">
                <Button size="lg" className="w-full sm:w-auto">
                  Get Started <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
              <Button
                variant="secondary"
                size="lg"
                className="w-full sm:w-auto">

                View Demo
              </Button>
            </div>
            <div className="mt-8 flex items-center space-x-6 text-sm text-gray-500">
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                HIPAA Aligned
              </div>
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                Zero Raw Data Transfer
              </div>
            </div>
          </motion.div>

          {/* Animated Hero Graphic */}
          <div className="relative h-[400px] bg-gray-50 rounded-2xl border border-gray-200 p-8 flex items-center justify-center overflow-hidden">
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>

            {/* Central Node */}
            <motion.div
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20"
              animate={{
                scale: [1, 1.05, 1]
              }}
              transition={{
                duration: 3,
                repeat: Infinity
              }}>

              <div className="w-24 h-24 bg-white rounded-full shadow-xl flex items-center justify-center border-4 border-indigo-100">
                <Cpu className="w-10 h-10 text-indigo-600" />
              </div>
              <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap font-semibold text-gray-700">
                Coordinator
              </div>
            </motion.div>

            {/* Hospital Nodes */}
            {[0, 1, 2].map((i) =>
            <motion.div
              key={i}
              className="absolute z-10"
              style={{
                top: i === 0 ? '20%' : i === 1 ? '80%' : '80%',
                left: i === 0 ? '50%' : i === 1 ? '20%' : '80%',
                transform: 'translate(-50%, -50%)'
              }}>

                <div className="w-16 h-16 bg-white rounded-xl shadow-lg flex items-center justify-center border border-gray-200">
                  <Activity className="w-8 h-8 text-teal-600" />
                </div>

                {/* Connection Line */}
                <svg
                className="absolute top-1/2 left-1/2 w-[200px] h-[200px] -z-10 pointer-events-none overflow-visible"
                style={{
                  transform: 'translate(-50%, -50%)'
                }}>

                  <motion.line
                  x1="50%"
                  y1="50%"
                  x2={i === 0 ? '50%' : i === 1 ? '150%' : '-50%'}
                  y2={i === 0 ? '150%' : '-50%'}
                  stroke="#E5E7EB"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                  initial={{
                    pathLength: 0
                  }}
                  animate={{
                    pathLength: 1
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity
                  }} />

                  <motion.circle
                  r="4"
                  fill="#4F46E5"
                  animate={{
                    cx: [i === 0 ? '50%' : i === 1 ? '150%' : '-50%', '50%'],
                    cy: [i === 0 ? '150%' : '-50%', '50%']
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: 'linear'
                  }} />

                </svg>
              </motion.div>
            )}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Enterprise-Grade Privacy Architecture
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Built for healthcare consortiums requiring mathematical privacy
              guarantees.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
            {
              icon: Database,
              title: 'Local-Only Retrieval',
              desc: 'Each hospital queries its own ChromaDB. No documents leave the premises.'
            },
            {
              icon: Shield,
              title: 'Differential Privacy',
              desc: 'Gaussian noise applied to embeddings before transmission. ε=1.0, δ=1e-5.'
            },
            {
              icon: FileText,
              title: 'Evidence-Based',
              desc: 'Structured answers with metadata citations. No raw patient text in responses.'
            },
            {
              icon: Cpu,
              title: 'Multi-Provider LLM',
              desc: 'Support for OpenRouter, Groq, Qwen, and OpenAI models.'
            }].
            map((feature, idx) =>
            <Card
              key={idx}
              className="hover:shadow-lg transition-shadow duration-300">

                <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-indigo-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {feature.desc}
                </p>
              </Card>
            )}
          </div>
        </div>
      </section>

      {/* How it Works Stepper */}
      <section id="how-it-works" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900">How It Works</h2>
          </div>

          <div className="relative">
            <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-100 -translate-y-1/2 hidden lg:block"></div>
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 relative z-10">
              {[
              {
                step: 1,
                title: 'Query',
                icon: Activity
              },
              {
                step: 2,
                title: 'Local Retrieval',
                icon: Database
              },
              {
                step: 3,
                title: 'DP Noise',
                icon: Shield
              },
              {
                step: 4,
                title: 'Aggregation',
                icon: GitMerge
              },
              {
                step: 5,
                title: 'Answer',
                icon: FileText
              }].
              map((item, idx) =>
              <div
                key={idx}
                className="flex flex-col items-center text-center group">

                  <div className="w-16 h-16 bg-white border-2 border-indigo-100 rounded-full flex items-center justify-center mb-4 shadow-sm group-hover:border-indigo-600 group-hover:scale-110 transition-all duration-300">
                    <item.icon className="w-6 h-6 text-indigo-600" />
                  </div>
                  <h3 className="font-bold text-gray-900 mb-1">{item.title}</h3>
                  <p className="text-xs text-gray-500">Step {item.step}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Security Section */}
      <section id="security" className="py-20 bg-indigo-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">
                Mathematically Guaranteed Privacy
              </h2>
              <p className="text-indigo-200 text-lg mb-8">
                Our architecture ensures that no raw text ever leaves the client
                nodes. We use differential privacy to add noise to embeddings,
                preventing reconstruction attacks.
              </p>
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-indigo-800/50 p-4 rounded-lg border border-indigo-700">
                  <div className="text-3xl font-bold text-teal-400 mb-1">
                    ε = 1.0
                  </div>
                  <div className="text-sm text-indigo-300">Privacy Budget</div>
                </div>
                <div className="bg-indigo-800/50 p-4 rounded-lg border border-indigo-700">
                  <div className="text-3xl font-bold text-teal-400 mb-1">
                    δ = 1e-5
                  </div>
                  <div className="text-sm text-indigo-300">Delta Parameter</div>
                </div>
              </div>
            </div>
            <div className="flex justify-center">
              <div className="relative">
                <div className="absolute inset-0 bg-teal-500 blur-3xl opacity-20 rounded-full"></div>
                <Shield
                  className="w-64 h-64 text-white relative z-10"
                  strokeWidth={1} />

                <Lock className="w-24 h-24 text-teal-400 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-50 py-12 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Product</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Security
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Pricing
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Documentation
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    API Reference
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Whitepaper
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Blog
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Contact
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Legal</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-indigo-600">
                    HIPAA BAA
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
            © 2026 FederatedMed. All rights reserved.
          </div>
        </div>
      </footer>
    </div>);

}