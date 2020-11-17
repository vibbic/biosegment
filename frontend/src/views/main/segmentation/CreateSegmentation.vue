<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Segmentation</div>
      </v-card-title>
      <v-card-text>
        <SegmentationForm :segmentation="newSegmentation"></SegmentationForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Segmentation, SegmentationUpdate, SegmentationCreate } from '@/api';
import { defaultSegmentation } from '@/interfaces';
import SegmentationForm from '@/components/SegmentationForm.vue';
import { dispatchCreateSegmentation } from '@/store/segmentation/actions';
import { filterUndefined } from '@/utils';

@Component({ components: { SegmentationForm } })
export default class CreateSegmentation extends Vue {
  public newSegmentation: SegmentationCreate = defaultSegmentation();
  public valid = false;

  public async mounted() {
    this.reset();
  }

  public reset() {
    this.newSegmentation = defaultSegmentation();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredSegmentation: SegmentationCreate = filterUndefined(this.newSegmentation);
      await dispatchCreateSegmentation(this.$store, filteredSegmentation);
      this.$router.push('/main/segmentations');
    }
  }
}
</script>
