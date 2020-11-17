<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Edit Segmentation</div>
      </v-card-title>
      <v-card-text>
        <SegmentationForm
          :segmentation="segmentationForm"
          title="Update Segmentation"
        ></SegmentationForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit">Save</v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Segmentation, SegmentationUpdate, SegmentationCreate } from '@/api';
import { defaultSegmentation } from '@/interfaces';
import SegmentationForm from '@/components/SegmentationForm.vue';
import {
  dispatchGetSegmentations,
  dispatchUpdateSegmentation,
} from '@/store/segmentation/actions';
import { component } from 'vue/types/umd';
import { readOneSegmentation } from '@/store/segmentation/getters';
import { filterUndefined, deepCopy } from '@/utils';

@Component({ components: { SegmentationForm } })
export default class EditSegmentation extends Vue {
  public segmentationForm: SegmentationUpdate = deepCopy(this.segmentation);
  public valid = false;

  public async mounted() {
    await dispatchGetSegmentations(this.$store);
    this.reset();
  }

  public reset() {
    this.segmentationForm = deepCopy(this.segmentation);
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredSegmentation: SegmentationUpdate = filterUndefined(this.segmentationForm);
      await dispatchUpdateSegmentation(this.$store, {
        id: this.segmentation.id,
        segmentation: filteredSegmentation,
      });
      this.$router.push('/main/segmentations');
    }
  }

  get segmentation() {
    return readOneSegmentation(this.$store)(+this.$router.currentRoute.params.id);
  }
}
</script>
