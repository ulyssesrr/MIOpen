#include "random.hpp"
#include <vector>

// RNN VANILLA configs
inline std::vector<int> get_rnn_num_layers() { return {{1, 3}}; }

inline std::vector<int> get_rnn_batchSize() { return {{1, 17}}; }

inline std::vector<int> get_rnn_seq_len() { return {{1, 3, 51}}; }

inline std::vector<int> get_rnn_vector_len() { return {31}; }

inline std::vector<int> get_rnn_hidden_size() { return {127}; }

// LSTM configs
inline std::vector<int> get_lstm_num_layers() { return {{1, 3}}; }

inline std::vector<int> get_lstm_batchSize() { return {{1, 17}}; }

inline std::vector<int> get_lstm_seq_len() { return {{1, 25}}; }

inline std::vector<int> get_lstm_vector_len() { return {17}; }

inline std::vector<int> get_lstm_hidden_size() { return {67}; }

// GRU configs
inline std::vector<int> get_gru_num_layers() { return {{1, 3}}; }

inline std::vector<int> get_gru_batchSize() { return {{1, 17}}; }

inline std::vector<int> get_gru_seq_len() { return {{1, 23}}; }

inline std::vector<int> get_gru_vector_len() { return {13}; }

inline std::vector<int> get_gru_hidden_size() { return {67}; }

inline std::vector<std::vector<int>> generate_batchSeq(const int batchSize, const int seqLength)
{

    int modval = 3;
    srand(modval);
    int currentval = batchSize;
    std::vector<int> batchSeq;
    for(int i = 0; i < seqLength; i++)
    {
        if(i > 0)
        {
            int nvalue = currentval - GET_RAND() % modval;
            currentval = (nvalue < 1) ? 1 : nvalue;
            // printf("current value: %d\n", currentval);
        }
        // printf("adding a value to batch sequence: %d\n", currentval);
        batchSeq.push_back(currentval);
    }
    return {batchSeq, {}};
}