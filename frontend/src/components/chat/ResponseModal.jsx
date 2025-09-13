import React, { useState, useEffect } from 'react';
import { X, Brain, MessageSquare, Loader2 } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { Badge } from '../ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ResponseModal = ({ isOpen, onClose, query }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen && query) {
      fetchAnalysis();
    }
  }, [isOpen, query]);

  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API}/chat`, { query });
      setAnalysis(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze query');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'angry':
      case 'frustrated':
      case 'sad':
      case 'disappointed':
        return 'bg-red-100 text-red-800';
      case 'curious':
      case 'excited':
      case 'grateful':
      case 'happy':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'p0':
        return 'bg-red-100 text-red-800';
      case 'p1':
        return 'bg-orange-100 text-orange-800';
      case 'p2':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold text-blue-600 flex items-center">
            <Brain className="h-5 w-5 mr-2" />
            AI Analysis & Response
          </DialogTitle>
        </DialogHeader>

        <div className="mt-4">
          <div className="bg-blue-50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-blue-800 mb-2">Query:</h3>
            <p className="text-blue-700">{query}</p>
          </div>

          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              <span className="ml-2 text-gray-600">Analyzing query...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {analysis && !loading && (
            <Tabs defaultValue="analysis" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="analysis" className="flex items-center">
                  <Brain className="h-4 w-4 mr-2" />
                  Internal Analysis
                </TabsTrigger>
                <TabsTrigger value="response" className="flex items-center">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Final Response
                </TabsTrigger>
              </TabsList>

              <TabsContent value="analysis" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg text-gray-800">Classification & Agent Analysis</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-600">Topic</label>
                        <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                          {analysis.classification?.topic || 'Unknown'}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-600">Sentiment</label>
                        <Badge className={getSentimentColor(analysis.classification?.sentiment)}>
                          {analysis.classification?.sentiment || 'Neutral'}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-600">Priority</label>
                        <Badge className={getPriorityColor(analysis.classification?.priority)}>
                          {analysis.classification?.priority || 'P2'}
                        </Badge>
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <label className="text-sm font-medium text-gray-600 block mb-2">Analysis Details</label>
                      <p className="text-gray-700 text-sm leading-relaxed">
                        {analysis.analysis_details || 'Query processed through multiple AI agents including classification and conversational analysis.'}
                      </p>
                      {analysis.agents_used && (
                        <div className="mt-2">
                          <span className="text-xs text-gray-500">Agents Used: </span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {analysis.agents_used.map((agent, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                {agent}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="response" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg text-gray-800">AI Generated Response</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-100">
                      <p className="text-gray-800 leading-relaxed whitespace-pre-line">
                        {analysis.response || 'No response generated.'}
                      </p>
                    </div>

                    <div className="mt-4 pt-4 border-t">
                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <span>Confidence Score: {analysis.confidence || 0}%</span>
                        <span>Generated in {analysis.processing_time || 0}ms</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ResponseModal;