#include <Types.h>
#include <Random.h>
#include <Array.h>
#include <ArrayDef.h>
#include <File.h>
#include <math/KahanSum.h>
#include <text/NumberText.h>
#include <BigFloat.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string.h>
#include <vector>

using namespace OUTER_NAMESPACE;
using namespace OUTER_NAMESPACE :: COMMON_LIBRARY_NAMESPACE ;
using namespace big_float;

constexpr uint64 LOG2_BLOCKS_PER_TASK = 7;
constexpr uint64 BLOCKS_PER_TASK = (uint64(1)<<LOG2_BLOCKS_PER_TASK);
constexpr uint64 LOG2_ALL_RESULTS_THRESHOLD = 7;
constexpr uint64 ALL_RESULTS_THRESHOLD = (uint64(1)<<LOG2_ALL_RESULTS_THRESHOLD);
static_assert(ALL_RESULTS_THRESHOLD <= 2*BLOCKS_PER_TASK, "The code for recording results only has a special case for sequenceOffset being zero.");
constexpr uint64 LOG2_RESULTS_PER_POWER_OF_2 = LOG2_ALL_RESULTS_THRESHOLD-1;
constexpr uint64 RESULTS_PER_POWER_OF_2 = (uint64(1)<<LOG2_RESULTS_PER_POWER_OF_2);
constexpr uint64 LOG2_ITEMS_PER_BLOCK = 20;
constexpr uint64 ITEMS_PER_BLOCK = (uint64(1)<<LOG2_ITEMS_PER_BLOCK);
constexpr uint64 LOG2_ITEMS_PER_TASK = LOG2_BLOCKS_PER_TASK + LOG2_ITEMS_PER_BLOCK;
constexpr uint64 ITEMS_PER_TASK = (uint64(1)<<LOG2_ITEMS_PER_TASK);
constexpr uint64 NUM_RESULTS_BEFORE_1_RESULT_PER_TASK = ALL_RESULTS_THRESHOLD + RESULTS_PER_POWER_OF_2*(LOG2_ITEMS_PER_TASK-LOG2_ALL_RESULTS_THRESHOLD+LOG2_ALL_RESULTS_THRESHOLD);
constexpr uint64 NUM_ITEMS_BEFORE_1_RESULT_PER_TASK = (uint64(1)<<(LOG2_ITEMS_PER_TASK+LOG2_ALL_RESULTS_THRESHOLD));
constexpr uint64 RANDOM_PER_ITEM = 4;
constexpr bool KEEP_LOOPING = true;

template<typename T>
struct OneNumberState {
	T sum;

	constexpr void reset() {
		sum = 0;
	}

	template<size_t FN>
	INLINE big_float::BigFloat<FN> asBigFloat() const {
		return big_float::BigFloat<FN>(sum);
	}

	INLINE void add(T item) {
		sum += item;
	}
};

template<typename T,size_t N>
struct TupleSumState {
	T sum[N];

	constexpr void reset() {
		for (size_t i = 0; i < N; ++i) {
			sum[i] = 0;
		}
	}
	template<size_t FN>
	big_float::BigFloat<FN> asBigFloat() const {
		// Add from smallest to largest
		big_float::BigFloat<FN> outSum = big_float::BigFloat<FN>(double(sum[N-1]));
		for (size_t i = N-2; i < N; --i) {
			outSum += big_float::BigFloat<FN>(double(sum[i]));
		}
		return outSum;
	}
};

// Original Kahan summation
template<typename T>
struct OrigKahanState : TupleSumState<T,2> {
	using TupleSumState<T,2>::sum;
	INLINE void add(T item) {
		const T localSumLow = item + sum[1];
		const T localSumHigh = sum[0] + localSumLow;
		sum[1] = localSumLow - (localSumHigh - sum[0]);
		sum[0] = localSumHigh;
	}
};

// Neumaier variation
template<typename T>
struct NeumaierState : TupleSumState<T,2> {
	using TupleSumState<T,2>::sum;
	INLINE void add(T item) {
		T localSumHigh = sum[0] + item;
		T localSumLow;
		if (std::abs(item) >= std::abs(sum[0])) {
			localSumLow = sum[0] - (localSumHigh - item);
		}
		else {
			localSumLow = item - (localSumHigh - sum[0]);
		}
		sum[0] = localSumHigh;
		sum[1] += localSumLow;
	}
};

// Klein variation
template<typename T>
struct KleinState : TupleSumState<T,3> {
	using TupleSumState<T,3>::sum;
	INLINE void add(T item) {
		T t = sum[0] + item;
		T c;
		if (std::abs(sum[0]) >= std::abs(item)) {
			c = (sum[0] - t) + item;
		}
		else {
			c = (item - t) + sum[0];
		}
		sum[0] = t;
		t = sum[1] + c;
		T cc;
		if (std::abs(sum[1]) >= std::abs(c)) {
			cc = (sum[1] - t) + c;
		}
		else {
			cc = (c - t) + sum[1];
		}
		sum[1] = t;
		sum[2] += cc;
	}
};

// New variation
template<typename T>
struct NewState : TupleSumState<T,2> {
	using TupleSumState<T,2>::sum;
	INLINE void add(T item) {
		math::kahanSumSingle(sum[0], sum[1], item);
	}
};

// Simple new variation
template<typename T>
struct SimpleNewState : TupleSumState<T,2> {
	using TupleSumState<T,2>::sum;
	INLINE void add(T item) {
		T localSumHigh = sum[0] + item;
		T localSumLow;
		if (std::abs(item) >= std::abs(sum[0])) {
			localSumLow = sum[0] - (localSumHigh - item);
		}
		else {
			localSumLow = item - (localSumHigh - sum[0]);
		}
		sum[0] = localSumHigh;
		sum[1] += localSumLow;

		// Add low part back into high part
		localSumHigh = sum[0] + sum[1];
		localSumLow = sum[1] - (localSumHigh - sum[0]);
		sum[0] = localSumHigh;
		sum[1] = localSumLow;
	}
};

template<typename T>
struct PairwiseState {
	std::vector<T> state;
	uint64 count;

	PairwiseState() {
		reset();
	}
	void reset() {
		state.clear();
		count = 0;
	}
	INLINE void add(T item) {
		size_t mask = 1;
		size_t bit = 0;
		while (mask & count) {
			item += state[bit];
			state[bit] = 0; // Clear just in case
			mask += mask;
			++bit;
		}
		if (bit >= state.size()) {
			state.push_back(item);
		}
		else {
			state[bit] = item;
		}
		++count;
		// Non-zero state values in indices corresponding with 1-bits of count
	}

	template<size_t FN>
	big_float::BigFloat<FN> asBigFloat() const {
		// Add from smallest to largest
		big_float::BigFloat<FN> outSum = big_float::constants::zero<FN>;
		for (size_t i = 0; i < state.size(); ++i) {
			if (count & (1ULL<<i)) {
				outSum += big_float::BigFloat<FN>(double(state[i]));
			}
		}
		return outSum;
	}
};

template<typename T>
struct BlockState : TupleSumState<T,2> {
	uint64 count;
	// blockMask of 0x3FF corresponds with a block size of 1024
	constexpr static uint64 blockMask = 0x3FF;

	using Parent = typename TupleSumState<T,2>;
	using Parent::sum;

	BlockState() {
		reset();
	}
	void reset() {
		Parent::reset();
		count = 0;
	}
	INLINE void add(T item) {
		sum[1] += item;
		if (!(count & blockMask)) {
			sum[0] += sum[1];
			sum[1] = 0;
		}
		++count;
	}
};

template<size_t N>
struct BigFloatState {
	big_float::BigFloat<N> sum;

	BigFloatState() {
		reset();
	}
	void reset() {
		sum = big_float::constants::zero<N>;
	}
	template<typename T>
	INLINE void add(T item) {
		sum += big_float::BigFloat<N>(double(item));
	}

	template<size_t FN>
	big_float::BigFloat<FN> asBigFloat() const {
		return big_float::BigFloat<FN>(sum);
	}
};

template<typename SUM_T>
void multiplyDoubleDouble(double a, double b, SUM_T& result) {
	// FIXME: Compare this approach to the approach splitting into 26 bits and 27 bits,
	//        with rounding (to get the 27 down to 26), which should be slightly more
	//        accurate.
	//        Casting to float will keep 24 bits, instead of 26.
	double aHigh = double(float(a));
	double aLow = a - aHigh;
	double bHigh = double(float(b));
	double bLow = b - bHigh;
	result += aLow*bLow;
	result += aLow*bHigh;
	result += aHigh*bLow;
	result += aHigh*bHigh;
}

template<typename SUM_T>
void multiplyDoubleDouble2(double a, double b, SUM_T& result) {
	// FIXME: Handle infinity, NaN, denormal!!!
	double aHigh = a;
	uint64& aHighInt = *(uint64*)(&aHigh);
	// Clear bottom bits
	aHighInt = aHighInt & ~uint64(0x3FFFFFF);
	// Round up
	aHighInt += aHighInt & uint64(0x4000000);
	double aLow = a - aHigh;
	double bHigh = b;
	uint64& bHighInt = *(uint64*)(&bHigh);
	// Clear bottom bits
	bHighInt = bHighInt & ~uint64(0x3FFFFFF);
	// Round up
	bHighInt += bHighInt & uint64(0x4000000);
	double bLow = b - bHigh;
	result += aLow*bLow;
	result += aLow*bHigh;
	result += aHigh*bLow;
	result += aHigh*bHigh;
}

enum class DataType {
	FLOAT,
	FLOAT2,
	DOUBLE,
	DOUBLE2,
	BIGFLOAT8
};

inline void convertToItemType(const BigFloat<8>& sample, float& item) {
	item = float(sample);
}
inline void convertToItemType(const BigFloat<8>& sample, double& item) {
	item = double(sample);
}
inline void convertToItemType(const BigFloat<8>& sample, Vec2f& item) {
	item[1] = float(sample);
	item[0] = float(sample - BigFloat<8>(item[1]));
}
inline void convertToItemType(const BigFloat<8>& sample, Vec2d& item) {
	item[1] = double(sample);
	item[0] = double(sample - BigFloat<8>(item[1]));
}
inline void convertToItemType(const BigFloat<8>& sample, BigFloat<8>& item) {
	item = sample;
}

template<typename STATE_T>
INLINE void sumSingle(STATE_T& state, const float& item) {
	state.add(item);
}
template<typename STATE_T>
INLINE void sumSingle(STATE_T& state, const double& item) {
	state.add(item);
}
template<typename STATE_T>
INLINE void sumSingle(STATE_T& state, const Vec2f& item) {
	state.add(item[0]);
	state.add(item[1]);
}
template<typename STATE_T>
INLINE void sumSingle(STATE_T& state, const Vec2d& item) {
	state.add(item[0]);
	state.add(item[1]);
}
template<typename STATE_T>
INLINE void sumSingle(STATE_T& state, const BigFloat<8>& item) {
	// Not sure why this case would be worth testing, but try converting to a few doubles.
	double v0 = double(item);
	BigFloat<8> removed(v0);
	BigFloat<8> b1(item - removed);
	double v1 = double(b1);
	removed += b1;
	BigFloat<8> b2(item - removed);
	double v2 = double(b2);
	removed += b2;
	BigFloat<8> b3(item - removed);
	double v3 = double(b3);
	state.add(v3);
	state.add(v2);
	state.add(v1);
	state.add(v0);
}
template<>
INLINE void sumSingle<BigFloatState<10>>(BigFloatState<10>& state, const BigFloat<8>& item) {
	state.sum += BigFloat<10>(item);
}

static bool shouldReportAfter(uint64 itemi) {
	if (itemi <= ALL_RESULTS_THRESHOLD) {
		return true;
	}
	// Report if all but the top 7 bits are zero.
	uint32 top_bit = BitScanR64(itemi);
	top_bit -= (LOG2_ALL_RESULTS_THRESHOLD-1);
	uint64 mask = ~(uint64(int64(-1)) << top_bit);
	return (itemi & mask) == 0;
}

static uint64 fileEntryToEndOffset(uint64 entry) {
	if (entry < ALL_RESULTS_THRESHOLD) {
		return entry + 1;
	}

	// FIXME: Verify this logic!!!
	if (entry < NUM_RESULTS_BEFORE_1_RESULT_PER_TASK) {
		entry -= (ALL_RESULTS_THRESHOLD/2);
		uint64 shift = entry >> (LOG2_ALL_RESULTS_THRESHOLD-1);
		entry &= (uint64(1) << (LOG2_ALL_RESULTS_THRESHOLD-1)) - 1;
		entry += (uint64(1) << (LOG2_ALL_RESULTS_THRESHOLD-1));
		uint64 offset = ((entry+1) << shift);
		assert(offset <= NUM_ITEMS_BEFORE_1_RESULT_PER_TASK);
		return offset;
	}
	return (entry - NUM_RESULTS_BEFORE_1_RESULT_PER_TASK + 1)*ITEMS_PER_TASK + NUM_ITEMS_BEFORE_1_RESULT_PER_TASK;
}

struct SequenceInfo {
	uint64 sequenceNumber;
	const char* summationMethodName;
	const char* methodDataTypeName;
	const char* itemDataTypeName;
	const char* distributionName;
};

static void makeFilename(
	Array<char>& filename,
	const char*const prefix,
	uint64 sequenceNumber,
	const char*const suffix
) {
	size_t prefixLength = strlen(prefix);
	size_t suffixLength = strlen(suffix);

	char seqNumText[17];
	sprintf(seqNumText, "%llX", sequenceNumber);
	size_t seqNumLength = strlen(seqNumText);

	filename.setSize(prefixLength + seqNumLength + suffixLength + 1);
	char* filenameData = filename.data();
	memcpy(filenameData, prefix, prefixLength);
	filenameData += prefixLength;
	memcpy(filenameData, seqNumText, seqNumLength);
	filenameData += seqNumLength;
	// Copy zero terminator, too.
	memcpy(filenameData, suffix, suffixLength + 1);
}

static void makeSumFilename(
	Array<char>& filename,
	const SequenceInfo& sequenceInfo,
	const char*const suffix
) {
	size_t summationMethodLength = strlen(sequenceInfo.summationMethodName);
	size_t methodDataTypeLength = strlen(sequenceInfo.methodDataTypeName);
	size_t itemDataTypeLength = strlen(sequenceInfo.itemDataTypeName);
	size_t distributionLength = strlen(sequenceInfo.distributionName);
	size_t suffixLength = strlen(suffix);

	char seqNumText[18];
	if (sequenceInfo.sequenceNumber != std::numeric_limits<uint64>::max()) {
		sprintf(seqNumText, "%llX", sequenceInfo.sequenceNumber);
	}
	else {
		seqNumText[0] = 'X';
		seqNumText[1] = 0;
	}
	size_t seqNumLength = strlen(seqNumText);

	filename.setSize(summationMethodLength + 1 + seqNumLength + 1 + methodDataTypeLength + 1 + itemDataTypeLength + 1 + distributionLength + suffixLength + 1);
	char* filenameData = filename.data();

	memcpy(filenameData, sequenceInfo.summationMethodName, summationMethodLength);
	filenameData += summationMethodLength;
	*filenameData = '_';
	++filenameData;

	memcpy(filenameData, seqNumText, seqNumLength);
	filenameData += seqNumLength;
	*filenameData = '_';
	++filenameData;

	memcpy(filenameData, sequenceInfo.methodDataTypeName, methodDataTypeLength);
	filenameData += methodDataTypeLength;
	*filenameData = '_';
	++filenameData;

	memcpy(filenameData, sequenceInfo.itemDataTypeName, itemDataTypeLength);
	filenameData += itemDataTypeLength;
	*filenameData = '_';
	++filenameData;

	memcpy(filenameData, sequenceInfo.distributionName, distributionLength);
	filenameData += distributionLength;

	// Copy zero terminator, too.
	memcpy(filenameData, suffix, suffixLength + 1);
}

template<typename STATE_T,typename ITEM_T>
bool analysisSingle(
	const SequenceInfo& baseSequenceInfo,
	const uint64 sequenceOffset,
	Array<BigFloat<10>>& methodSums,
	Array<BigFloat<10>>& targetSums,
	Array<BigFloat<10>>& absDiffs,
	Array<BigFloat<10>>& relDiffs,
	FILE* absDiffOutputFile,
	FILE* relDiffOutputFile
) {
	const size_t numSequences = (methodSums.size() < targetSums.size()) ? methodSums.size() : targetSums.size();
	assert(numSequences > 0);

	absDiffs.setSize(numSequences);
	relDiffs.setSize(numSequences);
	for (size_t i = 0; i < numSequences; ++i) {
		absDiffs[i] = methodSums[i] - targetSums[i];
		absDiffs[i].negative = false;
		BigFloat<10> magnitude = targetSums[i];
		magnitude.negative = false;
		relDiffs[i] = absDiffs[i] / magnitude;
	}
	std::sort(absDiffs.begin(), absDiffs.end());
	std::sort(relDiffs.begin(), relDiffs.end());
	std::sort(targetSums.begin(), targetSums.end());

	constexpr size_t numPercentiles = 11;
	constexpr double percentiles[numPercentiles] = {
		0.0,
		0.01,
		0.05,
		0.1,
		0.25,
		0.5,
		0.75,
		0.9,
		0.95,
		0.99,
		1.0
	};

	if (sequenceOffset == 1) {
		fprintf(absDiffOutputFile, "seqOff\tnumSeq\tmin\t1%%\t5%%\t10%%\t25%%\t50%%\t75%%\t90%%\t95%%\t99%%\tmax\n");
		fprintf(relDiffOutputFile, "seqOff\tnumSeq\tmin\t1%%\t5%%\t10%%\t25%%\t50%%\t75%%\t90%%\t95%%\t99%%\tmax\n");
	}
	fprintf(absDiffOutputFile, "%llu\t%llu", (unsigned long long)sequenceOffset, (unsigned long long)numSequences);
	fprintf(relDiffOutputFile, "%llu\t%llu", (unsigned long long)sequenceOffset, (unsigned long long)numSequences);

	for (size_t percentilei = 0; percentilei < numPercentiles; ++percentilei) {
		double t = (numSequences-1)*percentiles[percentilei];
		size_t index0 = size_t(t);
		size_t index1 = index0+1;
		t -= index0;
		if (index1 >= numSequences) {
			index1 = numSequences-1;
			t = 0;
		}

		double absDiff = double(absDiffs[index0]) + t*double(absDiffs[index1]-absDiffs[index0]);
		double relDiff = double(relDiffs[index0]) + t*double(relDiffs[index1]-relDiffs[index0]);
		double targetSum = double(targetSums[index0]) + t*double(targetSums[index1]-targetSums[index0]);

		fprintf(absDiffOutputFile, "\t%e", absDiff);
		fprintf(relDiffOutputFile, "\t%e", relDiff);
	}
	fprintf(absDiffOutputFile, "\n");
	fprintf(relDiffOutputFile, "\n");

	return true;
}

template<typename STATE_T,typename ITEM_T>
bool analysis(const SequenceInfo& baseSequenceInfo) {
	BufArray<char,128> filename;
	makeSumFilename(filename, baseSequenceInfo, "_AbsDiff.txt");
	FILE* absDiffOutputFile = fopen(filename.data(), "wb");
	if (absDiffOutputFile == nullptr) {
		printf("Unable to create file \"%s\" for writing.", filename.data());
		fflush(stdout);
		return false;
	}
	makeSumFilename(filename, baseSequenceInfo, "_RelDiff.txt");
	FILE* relDiffOutputFile = fopen(filename.data(), "wb");
	if (relDiffOutputFile == nullptr) {
		fclose(absDiffOutputFile);
		printf("Unable to create file \"%s\" for writing.", filename.data());
		fflush(stdout);
		return false;
	}

	Array<Array<BigFloat<10>>> methodSums;
	Array<Array<BigFloat<10>>> targetSums;

	Array<char> methodContents;
	Array<char> targetContents;
	for (uint64 sequenceNumber = 0; ; ++sequenceNumber) {
		SequenceInfo sequenceInfo(baseSequenceInfo);
		sequenceInfo.sequenceNumber = sequenceNumber;

		makeSumFilename(filename, sequenceInfo, "_Sum.bin");

		bool success = ReadWholeFile(filename.data(), methodContents);
		if (!success) {
			break;
		}
		size_t numCurrentMethodSums = (methodContents.size()/sizeof(STATE_T));
		if (numCurrentMethodSums != 0) {
			--numCurrentMethodSums;
		}
		if (numCurrentMethodSums == 0) {
			printf("Sum file \"%s\" is less than %u bytes long.", filename.data(), 2*(unsigned int)sizeof(STATE_T));
			fflush(stdout);
			return false;
		}
		const STATE_T*const currentMethodSums = (const STATE_T*)methodContents.data();


		sequenceInfo.methodDataTypeName = "double";
		sequenceInfo.summationMethodName = "BigFloat10";
		makeSumFilename(filename, sequenceInfo, "_Sum.bin");

		success = ReadWholeFile(filename.data(), targetContents);
		if (!success) {
			break;
		}
		size_t numCurrentTargetSums = (targetContents.size()/sizeof(BigFloat<10>));
		if (numCurrentTargetSums != 0) {
			--numCurrentTargetSums;
		}
		if (numCurrentTargetSums == 0) {
			printf("Sum file \"%s\" is less than %u bytes long.", filename.data(), 2*(unsigned int)sizeof(BigFloat<10>));
			fflush(stdout);
			return false;
		}
		const BigFloat<10>*const currentTargetSums = (const BigFloat<10>*)targetContents.data();

		size_t numCurrentSums = (numCurrentMethodSums < numCurrentTargetSums) ? numCurrentMethodSums : numCurrentTargetSums;
		if (methodSums.size() < numCurrentSums) {
			methodSums.setSize(numCurrentSums);
		}
		if (targetSums.size() < numCurrentSums) {
			targetSums.setSize(numCurrentSums);
		}
		for (size_t resulti = 0; resulti < numCurrentSums; ++resulti) {
			methodSums[resulti].append(currentMethodSums[resulti].asBigFloat<10>());
			targetSums[resulti].append(currentTargetSums[resulti]);
		}
	}

	BufArray<BigFloat<10>, 128> absDiffs;
	BufArray<BigFloat<10>, 128> relDiffs;
	for (size_t resulti = 0; resulti < methodSums.size(); ++resulti) {
		Array<BigFloat<10>>& methodSamples = methodSums[resulti];
		Array<BigFloat<10>>& targetSamples = targetSums[resulti];
		if (methodSamples.size() < 2) {
			continue;
		}
		assert(methodSamples.size() > 1 && targetSamples.size() > 1);

		bool success = analysisSingle<STATE_T,ITEM_T>(
			baseSequenceInfo, fileEntryToEndOffset(resulti),
			methodSamples, targetSamples, absDiffs, relDiffs,
			absDiffOutputFile, relDiffOutputFile);
		if (!success) {
			fclose(absDiffOutputFile);
			fclose(relDiffOutputFile);
			return false;
		}
	}

	fclose(absDiffOutputFile);
	fclose(relDiffOutputFile);

	return true;
}

static void setupRNG(Random256& rng, uint64 sequenceNumber) {
	// Set up random number generator from scratch
	rng.reseed(12345);
	for (uint64 i = 0; i < sequenceNumber; ++i) {
		rng.jump();
	}
}

template<typename STATE_T,typename ITEM_T>
bool sumTaskWrapper1(
	const std::function<void (BigFloat<8>&v)>& inverseCDF,
	const SequenceInfo& sequenceInfo
) {
	if (sequenceInfo.sequenceNumber == std::numeric_limits<uint64>::max()) {
		return analysis<STATE_T,ITEM_T>(sequenceInfo);
	}

	BufArray<char,128> sumFilename;

	// Load previous sum from file
	makeSumFilename(sumFilename, sequenceInfo, "_Sum.bin");

	ReadWriteFileHandle sumFile = OpenFileReadWrite(sumFilename.data());
	if (sumFile.isClear()) {
		printf("Failed to open sum file \"%s\".", sumFilename.data());
		fflush(stdout);
		return false;
	}

	BufArray<char,128> rngFilename;
	makeFilename(rngFilename, "RNG_", sequenceInfo.sequenceNumber, ".bin");

	ReadWriteFileHandle rngFile = OpenFileReadWrite(rngFilename.data());
	if (rngFile.isClear()) {
		printf("Failed to open RNG file \"%s\".", rngFilename.data());
		fflush(stdout);
		return false;
	}

	STATE_T state;
	Random256 rng;

	const uint64 sumFileSize = GetFileSize(sumFile);
	const uint64 rngFileSize = GetFileSize(rngFile);

	// Assume the last entry is invalid, in case it was written incorrectly.
	uint64 numValidSumEntries = (sumFileSize == 0) ? 0 : (sumFileSize-1)/sizeof(STATE_T);
	uint64 numValidRNGEntries = (rngFileSize == 0) ? 0 : (rngFileSize-1)/sizeof(rng);
	if (numValidRNGEntries < numValidSumEntries) {
		// RNG file must not have been written correctly,
		// so start from earlier sum.
		numValidSumEntries = numValidRNGEntries;
	}

	if (numValidSumEntries == 0) {
		state.reset();
		setupRNG(rng, sequenceInfo.sequenceNumber);
	}
	else {
		const uint64 lastValidSumStart = (numValidSumEntries - 1)*sizeof(STATE_T);
		bool success = SetFileOffset(sumFile, lastValidSumStart);
		if (!success) {
			printf("Failed to set the file offset to %llu in the sum file \"%s\".", lastValidSumStart, sumFilename.data());
			fflush(stdout);
			return false;
		}

		size_t numBytesRead = ReadFile(sumFile, &state, sizeof(state));
		if (numBytesRead != sizeof(state)) {
			printf("Failed to read the 2nd-last state from the sum file \"%s\".", sumFilename.data());
			fflush(stdout);
			return false;
		}

		// NOTE: NOT reading from last valid RNG entry; just the one corresponding with the last valid sum entry.
		const uint64 correspondingRNGStart = (numValidSumEntries - 1)*sizeof(rng);
		success = SetFileOffset(rngFile, correspondingRNGStart);
		if (!success) {
			printf("Failed to set the file offset to %llu in the RNG file \"%s\".", correspondingRNGStart, rngFilename.data());
			fflush(stdout);
			return false;
		}

		numBytesRead = ReadFile(rngFile, &rng, sizeof(rng));
		if (numBytesRead != sizeof(rng)) {
			printf("Failed to read the RNG state from the RNG file \"%s\".", rngFilename.data());
			fflush(stdout);
			return false;
		}

		const uint64 lastValidRNGEnd = numValidRNGEntries*sizeof(rng);
		success = SetFileOffset(rngFile, lastValidRNGEnd);
		if (!success) {
			printf("Failed to set the file offset to %llu in the RNG file \"%s\".", lastValidRNGEnd, rngFilename.data());
			fflush(stdout);
			return false;
		}
	}

	Array<ITEM_T> items;
	items.setSize(ITEMS_PER_BLOCK);

	uint64 sequenceOffset = (numValidSumEntries == 0) ? 0 : fileEntryToEndOffset(numValidSumEntries-1);
	printf("Starting sequence sum %llu at offset %llu, (%u M)\n", sequenceInfo.sequenceNumber, sequenceOffset, (unsigned int)(sequenceOffset>>20));
	fflush(stdout);
	uint64 task = (sequenceOffset/ITEMS_PER_TASK);
	uint64 startBlock = (sequenceOffset%ITEMS_PER_TASK)/ITEMS_PER_BLOCK;
	uint64 startItem = (sequenceOffset%ITEMS_PER_TASK)%ITEMS_PER_BLOCK;
	printf("Start task: %llu\nStart block: %llu\nStart item: %llu\n", task, startBlock, startItem);
	fflush(stdout);
	do {
		for (uint64 blocki = startBlock; blocki < BLOCKS_PER_TASK; ++blocki) {
			for (uint64 itemi = startItem; itemi < ITEMS_PER_BLOCK; ++itemi) {
				// Generate array of ITEMS_PER_BLOCK numbers
				uint64 sampleBits[RANDOM_PER_ITEM];
				for (uint64 i = 0; i < RANDOM_PER_ITEM; ++i) {
					sampleBits[i] = rng.next();
				}

				BigFloat<8> sample(sampleBits[0]);
				for (uint64 i = 1; i < RANDOM_PER_ITEM; ++i) {
					sample <<= 64;
					sample += BigFloat<8>(sampleBits[i]);
				}
				// Divide to range [0, 1)
				sample >>= 64*RANDOM_PER_ITEM;

				inverseCDF(sample);

				convertToItemType(sample, items[itemi]);
			}

			if (task == 0) {
				for (uint64 itemi = startItem; itemi < ITEMS_PER_BLOCK; ++itemi) {
					sumSingle(state, items[itemi]);
					++sequenceOffset;
					if (shouldReportAfter(sequenceOffset) || ((itemi == ITEMS_PER_BLOCK-1) && (blocki == BLOCKS_PER_TASK-1))) {
						size_t numBytesWritten = WriteFile(sumFile, &state, sizeof(state));
						if (numBytesWritten != sizeof(state)) {
							printf("Failed to write sum file \"%s\".", sumFilename.data());
							fflush(stdout);
							return false;
						}
						FlushFile(sumFile);
						++numValidSumEntries;

						if (numValidSumEntries > numValidRNGEntries) {
							numBytesWritten = WriteFile(rngFile, &rng, sizeof(rng));
							if (numBytesWritten != sizeof(rng)) {
								printf("Failed to write RNG file \"%s\".", rngFilename.data());
								fflush(stdout);
								return false;
							}
							FlushFile(rngFile);
							++numValidRNGEntries;
						}
					}
				}
			}
			else {
				// Don't need to check for reporting every item if we're
				// far enough along in the sequence.
				for (uint64 itemi = startItem; itemi < ITEMS_PER_BLOCK; ++itemi) {
					sumSingle(state, items[itemi]);
				}
				sequenceOffset += ITEMS_PER_BLOCK;
				if (shouldReportAfter(sequenceOffset) || (blocki == BLOCKS_PER_TASK-1)) {
					size_t numBytesWritten = WriteFile(sumFile, &state, sizeof(state));
					if (numBytesWritten != sizeof(state)) {
						printf("Failed to write sum file \"%s\".", sumFilename.data());
						fflush(stdout);
						return false;
					}
					FlushFile(sumFile);
					++numValidSumEntries;
					printf("%u M numbers added (result #%llu)\n", (unsigned int)(sequenceOffset>>20), numValidSumEntries-1);
					fflush(stdout);

					if (numValidSumEntries > numValidRNGEntries) {
						numBytesWritten = WriteFile(rngFile, &rng, sizeof(rng));
						if (numBytesWritten != sizeof(rng)) {
							printf("Failed to write RNG file \"%s\".", rngFilename.data());
							fflush(stdout);
							return false;
						}
						FlushFile(rngFile);
						++numValidRNGEntries;
					}
				}
			}
			startItem = 0;
		}
		startBlock = 0;
		++task;
	} while (KEEP_LOOPING);

	return true;
}
template<typename STATE_T>
bool sumTaskWrapper0(
	const DataType itemDataType,
	const std::function<void (BigFloat<8>&v)>& inverseCDF,
	const SequenceInfo& sequenceInfo
) {
	switch (itemDataType) {
		case DataType::FLOAT: {
			return sumTaskWrapper1<STATE_T,float>(inverseCDF, sequenceInfo);
		}
		case DataType::FLOAT2: {
			return sumTaskWrapper1<STATE_T,Vec2f>(inverseCDF, sequenceInfo);
		}
		case DataType::DOUBLE: {
			return sumTaskWrapper1<STATE_T,double>(inverseCDF, sequenceInfo);
		}
		case DataType::DOUBLE2: {
			return sumTaskWrapper1<STATE_T,Vec2d>(inverseCDF, sequenceInfo);
		}
		case DataType::BIGFLOAT8: {
			return sumTaskWrapper1<STATE_T,BigFloat<8>>(inverseCDF, sequenceInfo);
		}
	}
	return false;
}


void testFileEntryOffsets() {
	uint64 numValidSumEntries = 0;
	uint64 sequenceOffset = 0;
	uint64 task = (sequenceOffset/ITEMS_PER_TASK);
	uint64 startBlock = (sequenceOffset%ITEMS_PER_TASK)/ITEMS_PER_BLOCK;
	uint64 startItem = (sequenceOffset%ITEMS_PER_TASK)%ITEMS_PER_BLOCK;
	while (sequenceOffset <= (uint64(1)<<44)) {
		for (uint64 blocki = startBlock; blocki < BLOCKS_PER_TASK; ++blocki) {
			if (task == 0) {
				for (uint64 itemi = startItem; itemi < ITEMS_PER_BLOCK; ++itemi) {
					++sequenceOffset;
					if (shouldReportAfter(sequenceOffset) || ((itemi == ITEMS_PER_BLOCK-1) && (blocki == BLOCKS_PER_TASK-1))) {
						uint64 computedOffset = fileEntryToEndOffset(numValidSumEntries);
						printf("%lld %llX %llX %s\n", numValidSumEntries, sequenceOffset, computedOffset, (sequenceOffset == computedOffset) ? "match" : "MISMATCH!!!");
						fflush(stdout);
						if (sequenceOffset != computedOffset) {
							return;
						}
						++numValidSumEntries;
					}
				}
			}
			else {
				// Don't need to check for reporting every item if we're
				// far enough along in the sequence.
				sequenceOffset += ITEMS_PER_BLOCK;
				if (shouldReportAfter(sequenceOffset) || (blocki == BLOCKS_PER_TASK-1)) {
					uint64 computedOffset = fileEntryToEndOffset(numValidSumEntries);
					printf("%lld %llX %llX %s\n", numValidSumEntries, sequenceOffset, computedOffset, (sequenceOffset == computedOffset) ? "match" : "MISMATCH!!!");
					fflush(stdout);
					if (sequenceOffset != computedOffset) {
						return;
					}
					++numValidSumEntries;
				}
			}
			startItem = 0;
		}
		startBlock = 0;
		++task;
	}
}


int main(int argc,char** argv) {

	//testFileEntryOffsets();
	//return 0;

	// Argument 0 is the process path.
	// Argument 1 is the summation method name.
	// Argument 2 is the random sequence number.
	// Argument 3 is the method data type. (float, double)
	// Argument 4 is the data to sum type name. (float, double, double2, BigFloat8)
	// Argument 5 is the data distribution name.
	//
	// Examples:
	// Summation.exe OneNumber 0 double double Uniform2
	// Summation.exe BigFloat10 0 double double Uniform2
	//
	// Argument 3 was going to be the starting offset within that sequence, as a multiple of 2^20 (0x100000),
	// but now, it automatically figures out where to continue from.
	if (argc < 4) {
		printf("Not enough command line arguments!");
		fflush(stdout);
		return -1;
	}
	if (argc > 7) {
		printf("Too many command line arguments!");
		fflush(stdout);
		return -1;
	}

	const char*const summationMethodName = argv[1];

	uint64 sequenceNumber;
	if (argv[2][0] == '*' && argv[2][1] == 0 && strcmp(summationMethodName,"RNG") != 0) {
		// "*" is an indication to analyse all output data for actual methods
		sequenceNumber = std::numeric_limits<uint64>::max();
	}
	else {
		size_t numCharsInSeqNumber = text::textToInteger<16,true,false>((const char*)argv[2], (const char*)nullptr, sequenceNumber);
		if (numCharsInSeqNumber == 0) {
			printf("Command line argument 2 is not a valid hexadecimal integer.");
			fflush(stdout);
			return -1;
		}
	}

#if 0
	if (strcmp(summationMethodName,"RNG") == 0) {

		do {
			// Just generate random numbers in the sequence, to reach the end, ignoring any sums.
			for (uint64 i = 0; i < BLOCKS_PER_TASK*ITEMS_PER_BLOCK*RANDOM_PER_ITEM; ++i) {
				rng.next();
			}
			sequenceOffset += BLOCKS_PER_TASK;

			// Save RNG file.
			BufArray<char,128> filename;
			makeFilename(filename, "RNG_", sequenceNumber, sequenceOffset, ".bin");

			bool success = WriteWholeFile(filename.data(), (const char*)&rng, sizeof(rng));
			if (!success) {
				printf("Failed to write RNG file \"%s\".", filename.data());
				fflush(stdout);
				return -1;
			}
		} while (KEEP_LOOPING);

		// Done.
		return 0;
	}
#endif

	if (argc < 6) {
		printf("Not enough command line arguments for non-RNG task!");
		fflush(stdout);
		return -1;
	}

	const char*const methodDataTypeName = argv[3];
	const char*const itemDataTypeName = argv[4];
	const char*const distributionName = argv[5];

	enum class Method {
		ONE_NUMBER,
		KAHAN,
		NEUMAIER,
		KLEIN,
		NEW_FULL,
		NEW_SIMPLE,
		PAIRWISE,
		BLOCK,
		BIGFLOAT10
	};
	// FIXME: Figure out what the distribution is for contributions proportional to 1/d^2 in unit disk!!!
	enum class Distribution {
		UNIFORM,    // [0,1]
		UNIFORM2,   // [0,1]^2
		UNIFORM3,   // [0,1]^3
		UNIFORM4,   // [0,1]^4
		UNIFORMN2,  // [0,1]^(-2)
		UNIFORMN3,  // [0,1]^(-3)
		UNIFORMN4,  // [0,1]^(-4)
		UNIFORM_PN, // [-1,1]
		UNIFORM_1_5,// [0,1.5]
		UNIFORM_PN_1_5,// [-1.5,1.5]
		NORMAL,     // Normal(0,1)
		CAUCHY,     // Chauchy(0,1)
		LOG_NORMAL  // LogNormal(1,1)
	};

	Method method;
	if (strcmp(summationMethodName,"OneNumber") == 0) {
		method = Method::ONE_NUMBER;
	}
	else if (strcmp(summationMethodName,"Kahan") == 0) {
		method = Method::KAHAN;
	}
	else if (strcmp(summationMethodName,"Neumaier") == 0) {
		method = Method::NEUMAIER;
	}
	else if (strcmp(summationMethodName,"Klein") == 0) {
		method = Method::KLEIN;
	}
	else if (strcmp(summationMethodName,"NewFull") == 0) {
		method = Method::NEW_FULL;
	}
	else if (strcmp(summationMethodName,"NewSimple") == 0) {
		method = Method::NEW_SIMPLE;
	}
	else if (strcmp(summationMethodName,"Pairwise") == 0) {
		method = Method::PAIRWISE;
	}
	else if (strcmp(summationMethodName,"Block") == 0) {
		method = Method::BLOCK;
	}
	else if (strcmp(summationMethodName,"BigFloat10") == 0) {
		method = Method::BIGFLOAT10;
	}
	else {
		printf("Unsupported summation method \"%s\"!", summationMethodName);
		fflush(stdout);
		return -1;
	}

	DataType methodDataType;
	if (strcmp(methodDataTypeName,"float") == 0) {
		methodDataType = DataType::FLOAT;
	}
	else if (strcmp(methodDataTypeName,"double") == 0) {
		methodDataType = DataType::DOUBLE;
	}
	else {
		printf("Unsupported method data type \"%s\"!", methodDataTypeName);
		fflush(stdout);
		return -1;
	}

	DataType itemDataType;
	if (strcmp(itemDataTypeName,"float") == 0) {
		itemDataType = DataType::FLOAT;
	}
	else if (strcmp(itemDataTypeName,"float2") == 0) {
		itemDataType = DataType::FLOAT2;
	}
	else if (strcmp(itemDataTypeName,"double") == 0) {
		itemDataType = DataType::DOUBLE;
	}
	else if (strcmp(itemDataTypeName,"double2") == 0) {
		itemDataType = DataType::DOUBLE2;
	}
	else if (strcmp(itemDataTypeName,"BigFloat8") == 0) {
		itemDataType = DataType::BIGFLOAT8;
	}
	else {
		printf("Unsupported item data type \"%s\"!", itemDataTypeName);
		fflush(stdout);
		return -1;
	}

	Distribution distribution;
	std::function<void (BigFloat<8>&v)> inverseCDF;
	if (strcmp(distributionName,"Uniform") == 0) {
		distribution = Distribution::UNIFORM;
		inverseCDF = [](BigFloat<8>&v) {};
	}
	else if (strcmp(distributionName,"Uniform2") == 0) {
		distribution = Distribution::UNIFORM2;
		inverseCDF = [](BigFloat<8>&v) {
			v *= v;
		};
	}
	else if (strcmp(distributionName,"Uniform3") == 0) {
		distribution = Distribution::UNIFORM3;
		inverseCDF = [](BigFloat<8>&v) {
			v *= (v*v);
		};
	}
	else if (strcmp(distributionName,"Uniform4") == 0) {
		distribution = Distribution::UNIFORM4;
		inverseCDF = [](BigFloat<8>&v) {
			v *= v;
			v *= v;
		};
	}
	else if (strcmp(distributionName,"UniformN2") == 0) {
		distribution = Distribution::UNIFORMN2;
		inverseCDF = [](BigFloat<8>&v) {
			v = constants::one<8> / v;
			v *= v;
		};
	}
	else if (strcmp(distributionName,"UniformN3") == 0) {
		distribution = Distribution::UNIFORMN3;
		inverseCDF = [](BigFloat<8>&v) {
			v = constants::one<8> / v;
			v *= (v*v);
		};
	}
	else if (strcmp(distributionName,"UniformN4") == 0) {
		distribution = Distribution::UNIFORMN4;
		inverseCDF = [](BigFloat<8>&v) {
			v = constants::one<8> / v;
			v *= v;
			v *= v;
		};
	}
	else if (strcmp(distributionName,"Uniform_PN") == 0) {
		distribution = Distribution::UNIFORM_PN;
		inverseCDF = [](BigFloat<8>&v) {
			v <<= 1;
			v -= constants::one<8>;
		};
	}
	else if (strcmp(distributionName,"Uniform_1_5") == 0) {
		distribution = Distribution::UNIFORM_1_5;
		inverseCDF = [](BigFloat<8>&v) {
			v *= BigFloat<8>(3);
			v >>= 1;
		};
	}
	else if (strcmp(distributionName,"Uniform_PN_1_5") == 0) {
		distribution = Distribution::UNIFORM_PN_1_5;
		inverseCDF = [](BigFloat<8>&v) {
			v <<= 1;
			v -= constants::one<8>;
			v *= BigFloat<8>(3);
			v >>= 1;
		};
	}
	else if (strcmp(distributionName,"Normal") == 0) {
		distribution = Distribution::NORMAL;
		// FIXME: Implement this!!!
	}
	else if (strcmp(distributionName,"Cauchy") == 0) {
		distribution = Distribution::CAUCHY;
		inverseCDF = [](BigFloat<8>&v) {
			v <<= 1;
			v -= constants::one<8>;
			v *= constants::tau<8>;
			v >>= 2;

			BigFloat<8> s;
			BigFloat<8> c;
			s.sin(v);
			c.cos(v);
			v = s/c;
		};
	}
	else if (strcmp(distributionName,"LogNormal") == 0) {
		distribution = Distribution::LOG_NORMAL;
		// FIXME: Implement this!!!
	}
	else {
		printf("Unsupported distribution \"%s\"!", distributionName);
		fflush(stdout);
		return -1;
	}

	SequenceInfo sequenceInfo;
	sequenceInfo.sequenceNumber = sequenceNumber;
	sequenceInfo.summationMethodName = summationMethodName;
	sequenceInfo.methodDataTypeName = methodDataTypeName;
	sequenceInfo.itemDataTypeName = itemDataTypeName;
	sequenceInfo.distributionName = distributionName;

	bool success = false;
	if (methodDataType == DataType::FLOAT) {
		switch (method) {
			case Method::ONE_NUMBER: {
				success = sumTaskWrapper0<OneNumberState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::KAHAN: {
				success = sumTaskWrapper0<OrigKahanState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEUMAIER: {
				success = sumTaskWrapper0<NeumaierState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::KLEIN: {
				success = sumTaskWrapper0<KleinState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEW_FULL: {
				success = sumTaskWrapper0<NewState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEW_SIMPLE: {
				success = sumTaskWrapper0<SimpleNewState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::PAIRWISE: {
				success = sumTaskWrapper0<PairwiseState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::BLOCK: {
				success = sumTaskWrapper0<BlockState<float>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::BIGFLOAT10: {
				success = sumTaskWrapper0<BigFloatState<10>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
		}
	}
	else {
		switch (method) {
			case Method::ONE_NUMBER: {
				success = sumTaskWrapper0<OneNumberState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::KAHAN: {
				success = sumTaskWrapper0<OrigKahanState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEUMAIER: {
				success = sumTaskWrapper0<NeumaierState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::KLEIN: {
				success = sumTaskWrapper0<KleinState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEW_FULL: {
				success = sumTaskWrapper0<NewState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::NEW_SIMPLE: {
				success = sumTaskWrapper0<SimpleNewState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::PAIRWISE: {
				success = sumTaskWrapper0<PairwiseState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::BLOCK: {
				success = sumTaskWrapper0<BlockState<double>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
			case Method::BIGFLOAT10: {
				success = sumTaskWrapper0<BigFloatState<10>>(itemDataType, inverseCDF, sequenceInfo);
				break;
			}
		}
	}

	return success ? 0 : -1;
}
