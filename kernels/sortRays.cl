
#define MAX_LOCAL_SIZE 256
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// needed helper methods
inline void swap(Ray *a, Ray *b) {
	Ray tmp;
	tmp = *b;
	*b = *a;
	*a = tmp;
}

// dir == 1 means descending
inline void sort(Ray *a, Ray *b, char dir) {
	if ((a->type < b->type) == dir) swap(a, b);
	/*if ((a->type == b->type && 
			(a->screenCoords.y == b->screenCoords.y && a->screenCoords.x > b->screenCoords.x) ||
			(a->screenCoords.y  > b->screenCoords.y)) || 
		(a->type < b->type) == dir) swap(a, b); // TODO!!!!!!!!!!!!!!!!*/
}

inline void swapLocal(__local Ray *a, __local Ray *b) {
	Ray tmp;
	tmp = *b;
	*b = *a;
	*a = tmp;
}

// dir == 1 means descending
inline void sortLocal(__local Ray *a, __local Ray *b, char dir) {
	if ((a->type < b->type) == dir) swapLocal(a, b); //TODO !!!!!!!!!!!!!!!!!!!!!!!
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void Sort_BitonicMergesortStart(const __global Ray* inArray, __global Ray* outArray, int shift)
{
	__local Ray local_buffer[MAX_LOCAL_SIZE * 2];
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);

	uint index = get_group_id(0) * (MAX_LOCAL_SIZE * 2) + lid;
	//load into local mem
	local_buffer[lid] = inArray[shift + index];
	local_buffer[lid + MAX_LOCAL_SIZE] = inArray[shift + index + MAX_LOCAL_SIZE];

	uint clampedGID = gid & (MAX_LOCAL_SIZE - 1);

	// bitonic merge
	for (uint blocksize = 2; blocksize < MAX_LOCAL_SIZE * 2; blocksize <<= 1) {
		char dir = (clampedGID & (blocksize / 2)) == 0; // sort every other block in the other direction (faster % calc)
#pragma unroll
		for (uint stride = blocksize >> 1; stride > 0; stride >>= 1){
			barrier(CLK_LOCAL_MEM_FENCE);
			uint idx = 2 * lid - (lid & (stride - 1)); //take every other input BUT starting neighbouring within one block
			sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
		}
	}

	// bitonic merge for biggest group is special (unrolling this so we dont need ifs in the part above)
	char dir = (clampedGID & 0); //even or odd? sort accordingly
#pragma unroll
	for (uint stride = MAX_LOCAL_SIZE; stride > 0; stride >>= 1){
		barrier(CLK_LOCAL_MEM_FENCE);
		uint idx = 2 * lid - (lid & (stride - 1));
		sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
	}

	// sync and write back
	barrier(CLK_LOCAL_MEM_FENCE);
	outArray[shift + index] = local_buffer[lid];
	outArray[shift + index + MAX_LOCAL_SIZE] = local_buffer[lid + MAX_LOCAL_SIZE];
}

__kernel void Sort_BitonicMergesortLocal(__global Ray* data, const uint size, const uint blocksize, uint stride, int shift)
{
	// This Kernel is basically the same as Sort_BitonicMergesortStart except of the "unrolled" part and the provided parameters
	__local Ray local_buffer[2 * MAX_LOCAL_SIZE];
	uint gid = get_global_id(0);
	uint groupId = get_group_id(0);
	uint lid = get_local_id(0);
	uint clampedGID = gid & (size / 2 - 1);

	uint index = groupId * (MAX_LOCAL_SIZE * 2) + lid;
	//load into local mem
	local_buffer[lid] = data[shift + index];
	local_buffer[lid + MAX_LOCAL_SIZE] = data[shift + index + MAX_LOCAL_SIZE];

	// bitonic merge
	char dir = (clampedGID & (blocksize / 2)) == 0; //same as above, % calc
#pragma unroll
	for (; stride > 0; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		uint idx = 2 * lid - (lid & (stride - 1));
		sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
	}

	// sync and write back
	barrier(CLK_LOCAL_MEM_FENCE);
	data[shift + index] = local_buffer[lid];
	data[shift + index + MAX_LOCAL_SIZE] = local_buffer[lid + MAX_LOCAL_SIZE];
}

__kernel void Sort_BitonicMergesortGlobal(__global Ray* data, const uint size, const uint blocksize, const uint stride, int shift)
{
	// TO DO: Kernel implementation
	uint gid = get_global_id(0);
	uint clampedGID = gid & (size / 2 - 1);

	//calculate index and dir like above
	uint index = 2 * clampedGID - (clampedGID & (stride - 1));
	char dir = (clampedGID & (blocksize / 2)) == 0; //same as above, % calc

	//bitonic merge
	Ray left = data[shift + index];
	Ray right = data[shift + index + stride];

	sort(&left, &right, dir);

	// writeback
	data[shift + index] = left;
	data[shift + index + stride] = right;
}