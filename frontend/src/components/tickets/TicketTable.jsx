import React, { useState, useEffect } from 'react';
import { Badge } from '../ui/badge';
import { Loader2, AlertCircle } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TicketTable = () => {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTickets();
  }, []);

  const fetchTickets = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/tickets`);
      setTickets(response.data.tickets || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch tickets');
      console.error('Tickets fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'p0':
      case 'high':
        return 'bg-red-100 text-red-800 hover:bg-red-200';
      case 'p1':
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200';
      case 'p2':
      case 'low':
        return 'bg-green-100 text-green-800 hover:bg-green-200';
      default:
        return 'bg-gray-100 text-gray-800 hover:bg-gray-200';
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'angry':
      case 'frustrated':
      case 'sad':
      case 'disappointed':
      case 'negative':
        return 'bg-red-100 text-red-800 hover:bg-red-200';
      case 'curious':
      case 'excited':
      case 'grateful':
      case 'happy':
      case 'positive':
        return 'bg-green-100 text-green-800 hover:bg-green-200';
      case 'neutral':
      case 'confused':
      case 'uncertain':
        return 'bg-gray-100 text-gray-800 hover:bg-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 hover:bg-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <div className="flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="ml-2 text-gray-600">Loading tickets...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <div className="flex items-center justify-center text-red-600">
          <AlertCircle className="h-8 w-8 mr-2" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Serial No.
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Ticket Number
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Topic
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Sentiment
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Priority
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Response
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {tickets.length === 0 ? (
              <tr>
                <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                  No tickets found
                </td>
              </tr>
            ) : (
              tickets.map((ticket, index) => (
                <tr
                  key={ticket.id || index}
                  className={`transition-colors duration-150 hover:bg-blue-50 ${
                    index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                  }`}
                >
                  <td className="px-6 py-4 text-sm text-gray-900 font-medium">
                    {index + 1}
                  </td>
                  <td className="px-6 py-4 text-sm font-mono text-blue-600 font-medium">
                    {ticket.ticketNumber || `TKT-2024-${String(index + 1).padStart(3, '0')}`}
                  </td>
                  <td className="px-6 py-4">
                    <Badge
                      variant="secondary"
                      className="bg-blue-100 text-blue-800 hover:bg-blue-200 transition-colors duration-150"
                    >
                      {ticket.topic || 'Unknown'}
                    </Badge>
                  </td>
                  <td className="px-6 py-4">
                    <Badge
                      variant="secondary"
                      className={`transition-colors duration-150 ${getSentimentColor(ticket.sentiment)}`}
                    >
                      {ticket.sentiment || 'Neutral'}
                    </Badge>
                  </td>
                  <td className="px-6 py-4">
                    <Badge
                      variant="secondary"
                      className={`transition-colors duration-150 ${getPriorityColor(ticket.priority)}`}
                    >
                      {ticket.priority || 'P2'}
                    </Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600 max-w-xs truncate">
                    {ticket.response || 'No response available'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TicketTable;